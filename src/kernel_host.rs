//! The Python shell's kernel host: the native kernel (the `orbital_kernel` extension) runs the program and answers
//! every ask, so the generated project is the program, the language it needs and thin FastAPI routes. A program that
//! uses a Python-only (machine-learning) operator, which the kernel does not evaluate, keeps the generated host.

use orbital_core_base::generate::GeneratedFile;
use orbital_core_base::ir::effects::OirEffect;
use orbital_core_base::ir::expr::OirExpr;
use orbital_core_base::ir::{OirModule, OirTrait};
use orbital_core_base::language::registry;
use orbital_core_base::str_utils::to_kebab_case_simple;
use std::collections::BTreeSet;

pub const HOST_OPTION: &str = "host";

/// Who runs the program's execution steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Host {
    Generated,
    Kernel,
}

/// An operator the kernel does not evaluate: its category targets Python alone (`target: ["python"]`).
fn is_python_only(name: &str) -> bool {
    registry::get_operator(name)
        .and_then(|op| registry::categories().get(&op.category))
        .is_some_and(|category| category.target.iter().all(|t| t == "python"))
}

fn expr_operators(expr: &OirExpr, out: &mut BTreeSet<String>) {
    match expr {
        OirExpr::Call { function, .. } => {
            out.insert(function.clone());
        }
        OirExpr::Effect { effect } => {
            if let OirEffect::Custom { name, .. } = effect.as_ref() {
                out.insert(name.clone());
            }
        }
        _ => {}
    }
    for child in expr.child_exprs() {
        expr_operators(child, out);
    }
}

fn trait_operators(trait_def: &OirTrait, out: &mut BTreeSet<String>) {
    let transitions = trait_def
        .transitions
        .iter()
        .flat_map(|t| t.guard.iter().chain(t.effects.iter()));
    let guards = trait_def.guards.iter().map(|g| &g.expression);
    let ticks = trait_def
        .ticks
        .iter()
        .flat_map(|t| t.guard.iter().chain(t.effects.iter()));
    let listeners = trait_def.listeners.iter().filter_map(|l| l.guard.as_ref());
    for expr in transitions.chain(guards).chain(ticks).chain(listeners) {
        expr_operators(expr, out);
    }
}

/// The Python-only operators the program uses, in name order.
pub fn python_only_operators(module: &OirModule) -> Vec<String> {
    let mut used = BTreeSet::new();
    for trait_def in &module.traits {
        trait_operators(trait_def, &mut used);
    }
    used.into_iter()
        .filter(|name| is_python_only(name))
        .collect()
}

/// The host for `requested` (`auto` | `kernel` | `generated`). `auto` is the kernel unless the program uses a
/// Python-only operator; `kernel` on such a program is refused, naming the operators.
pub fn choose(requested: &str, module: &OirModule) -> Result<Host, String> {
    let python_only = python_only_operators(module);
    match requested {
        "generated" => Ok(Host::Generated),
        "auto" => Ok(if python_only.is_empty() { Host::Kernel } else { Host::Generated }),
        "kernel" if python_only.is_empty() => Ok(Host::Kernel),
        "kernel" => Err(format!(
            "host=kernel cannot run this program: the kernel does not evaluate the Python-only operator(s) {}; use host=generated",
            python_only.join(", ")
        )),
        other => Err(format!("host must be auto, kernel or generated, got {other}")),
    }
}

/// The routes module of a kernel-hosted app: one `POST /<trait>/events` per trait, the contract the generated host
/// serves, answered by `core.kernel_host`.
fn routes(module: &OirModule) -> String {
    let mut traits = String::new();
    for trait_def in &module.traits {
        traits.push_str(&format!(
            "    \"{}\": ({:?}, {:?}),\n",
            to_kebab_case_simple(&trait_def.name),
            trait_def.orbital_name,
            trait_def.name
        ));
    }
    format!(
        "\"\"\"Routes of the kernel-hosted app: the program runs in the native kernel, which answers its own I/O.\"\"\"\n\n\
         from ...core.kernel_host import register_kernel_routes\n\n\
         TRAITS = {{\n{traits}}}\n\n\n\
         def register_routes(app):\n    register_kernel_routes(app, TRAITS)\n"
    )
}

/// What a kernel-hosted project holds under `generated/`: the resolved program, the language it runs against and the
/// routes. `program` and `language` are the JSON texts.
pub fn files(module: &OirModule, program: &str, language: &str) -> Vec<GeneratedFile> {
    vec![
        GeneratedFile::new("program.json", program.to_string()),
        GeneratedFile::new("language.json", language.to_string()),
        GeneratedFile::new("routes/__init__.py", routes(module)),
        GeneratedFile::new(
            "__init__.py",
            "\"\"\"Generated Orbital code: the program, its language and the routes the kernel host serves.\"\"\"\n\nfrom . import routes\n\n__all__ = [\"routes\"]\n".to_string(),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use orbital_core_base::ir::{OirTrait, OirTransition};

    fn module_with(function: &str) -> OirModule {
        let mut trait_def = OirTrait::new("Mixer");
        trait_def.orbital_name = "Lab".to_string();
        trait_def.transitions.push(
            OirTransition::new("idle", "idle", "GO")
                .with_effects(vec![OirExpr::call(function, vec![])]),
        );
        let mut module = OirModule::new("app", "1.0.0");
        module.traits.push(trait_def);
        module
    }

    #[test]
    fn a_program_with_a_python_only_operator_keeps_the_generated_host() {
        let module = module_with("nn/linear");
        assert_eq!(
            python_only_operators(&module),
            vec!["nn/linear".to_string()]
        );
        assert_eq!(choose("auto", &module), Ok(Host::Generated));
        assert!(choose("kernel", &module).unwrap_err().contains("nn/linear"));
        assert_eq!(choose("generated", &module), Ok(Host::Generated));
    }

    #[test]
    fn control_a_program_with_only_shared_operators_runs_on_the_kernel() {
        let module = module_with("+");
        assert!(python_only_operators(&module).is_empty());
        assert_eq!(choose("auto", &module), Ok(Host::Kernel));
        assert_eq!(choose("kernel", &module), Ok(Host::Kernel));
        assert_eq!(choose("generated", &module), Ok(Host::Generated));
        assert!(choose("maybe", &module).is_err());
    }

    #[test]
    fn the_routes_name_each_trait_by_its_kebab_path_and_its_orbital() {
        let code = routes(&module_with("+"));
        assert!(code.contains("\"mixer\": (\"Lab\", \"Mixer\"),"), "{code}");
        assert!(
            code.contains("register_kernel_routes(app, TRAITS)"),
            "{code}"
        );
    }
}
