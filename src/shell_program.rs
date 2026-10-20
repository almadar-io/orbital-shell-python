//! `orb-shell-python`: the Python shell as a program (`orbital_core::shell_protocol`).

use crate::kernel_host::{self, Host, HOST_OPTION};
use crate::{KernelProject, PythonBackend};
use orbital_core_base::project::write_project;
use orbital_core_base::shell_protocol::{
    description, written_files, GenerateRequest, GenerateResponse, ShellDescription, ShellFailure,
    ShellOption, ShellProgram, CODE_BAD_REQUEST, CODE_GENERATE_FAILED,
};
use std::path::Path;

pub struct PythonProgram;

/// The native kernel extension, shipped beside the shell program and copied into a kernel-hosted project.
const KERNEL_EXTENSION: &str = if cfg!(windows) {
    "orbital_kernel.pyd"
} else {
    "orbital_kernel.abi3.so"
};

fn host_option() -> ShellOption {
    ShellOption {
        name: HOST_OPTION.to_string(),
        description: "Who runs the program: the native kernel, or the generated Python handlers. auto is the kernel unless the program uses a Python-only (ML) operator, the one program class that stays on the generated handlers".to_string(),
        values: Some(vec!["auto".to_string(), "kernel".to_string(), "generated".to_string()]),
        default: Some("auto".to_string()),
    }
}

fn requested_host(request: &GenerateRequest) -> Result<&str, ShellFailure> {
    match request.options.get(HOST_OPTION) {
        None => Ok("auto"),
        Some(serde_json::Value::String(s)) => Ok(s),
        Some(other) => Err(ShellFailure::new(
            CODE_BAD_REQUEST,
            format!("option {HOST_OPTION} must be a string, got {other}"),
        )),
    }
}

fn kernel_project(request: &GenerateRequest) -> Result<KernelProject, ShellFailure> {
    let resolved = request.resolved.as_ref().ok_or_else(|| {
        ShellFailure::new(
            CODE_BAD_REQUEST,
            "the kernel host needs the resolved program, and the request carries none",
        )
    })?;
    let program = serde_json::to_string(resolved).map_err(|e| {
        ShellFailure::new(CODE_GENERATE_FAILED, format!("serialize the program: {e}"))
    })?;
    let language = orbital_core_base::language_data::language_bundle_json()
        .map_err(|e| ShellFailure::new(CODE_GENERATE_FAILED, format!("{}: {e}", e.code())))?;
    Ok(KernelProject { program, language })
}


impl ShellProgram for PythonProgram {
    fn describe(&self) -> ShellDescription {
        description(
            "python",
            env!("CARGO_PKG_VERSION"),
            true,
            vec![host_option()],
        )
    }

    fn generate(
        &self,
        request: &GenerateRequest,
        output: &Path,
    ) -> Result<GenerateResponse, ShellFailure> {
        let host = kernel_host::choose(requested_host(request)?, &request.module)
            .map_err(|e| ShellFailure::new(CODE_BAD_REQUEST, e))?;
        let mut backend = PythonBackend::new();
        if host == Host::Kernel {
            backend.kernel = Some(kernel_project(request)?);
        }
        let generated = write_project(
            &request.module,
            output,
            &crate::PythonShell::new(),
            &backend,
            &request.write,
        )?;
        let mut warnings = generated.warnings.clone();
        if host == Host::Kernel {
            warnings.extend(orbital_core_base::shell_protocol::ship_kernel_beside_program(KERNEL_EXTENSION, output)?);
        } else {
            warnings.extend(kernel_host::generated_host_note(&request.module));
        }
        Ok(GenerateResponse {
            files: written_files(&generated),
            warnings,
            next_steps: vec![
                format!("cd {}", output.display()),
                "python -m venv .venv && source .venv/bin/activate".to_string(),
                "pip install -e .".to_string(),
                "python -m orbital_app.main".to_string(),
            ],
        })
    }

    fn emit(
        &self,
        request: &GenerateRequest,
    ) -> Result<orbital_core_base::generate::GeneratedOutput, ShellFailure> {
        use orbital_core_base::generate::Backend as _;
        Ok(crate::PythonBackend::new().generate(&request.module))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use orbital_core_base::ir::expr::OirExpr;
    use orbital_core_base::ir::{OirModule, OirTrait, OirTransition};
    use orbital_core_base::shell_protocol::IR_VERSION;

    fn request(function: &str, host: &str) -> GenerateRequest {
        let mut trait_def = OirTrait::new("Mixer");
        trait_def.transitions.push(
            OirTransition::new("idle", "idle", "GO")
                .with_effects(vec![OirExpr::call(function, vec![])]),
        );
        let mut module = OirModule::new("app", "1.0.0");
        module.traits.push(trait_def);
        GenerateRequest {
            ir_version: IR_VERSION,
            module,
            resolved: None,
            source_path: None,
            write: Default::default(),
            options: [(
                HOST_OPTION.to_string(),
                serde_json::Value::String(host.to_string()),
            )]
            .into_iter()
            .collect(),
        }
    }

    #[test]
    fn the_shell_asks_for_the_resolved_program_and_offers_the_host_option() {
        let description = PythonProgram.describe();
        assert!(description.needs_resolved);
        let option = description
            .options
            .iter()
            .find(|o| o.name == HOST_OPTION)
            .unwrap();
        assert_eq!(option.default.as_deref(), Some("auto"));
    }

    #[test]
    fn a_kernel_host_for_a_program_with_a_python_only_operator_is_refused_by_name() {
        let out = std::env::temp_dir().join(format!("orb-py-host-{}", std::process::id()));
        let failure = PythonProgram
            .generate(&request("nn/linear", "kernel"), &out)
            .unwrap_err();
        assert_eq!(failure.code, CODE_BAD_REQUEST);
        assert!(failure.message.contains("nn/linear"), "{}", failure.message);
    }

    #[test]
    fn control_a_kernel_host_without_the_resolved_program_is_refused() {
        let out = std::env::temp_dir().join(format!("orb-py-host-none-{}", std::process::id()));
        let failure = PythonProgram
            .generate(&request("+", "kernel"), &out)
            .unwrap_err();
        assert_eq!(failure.code, CODE_BAD_REQUEST);
        assert!(
            failure.message.contains("resolved program"),
            "{}",
            failure.message
        );
    }
}
