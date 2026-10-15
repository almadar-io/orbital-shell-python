//! OIR Expression to Python Compiler
//!
//! Compiles OirExpr to Python code, with special handling for neural network
//! and tensor operations that compile to PyTorch.

use orbital_core::ir::{
    OirBinaryOp, OirBinding, OirBindingRoot, OirEffect, OirExpr, OirPattern, OirUnaryOp,
};
use orbital_core::str_utils::escape_python_string;
use std::collections::HashMap;

/// Context for code generation
#[derive(Debug, Clone, Default)]
pub struct CodegenContext {
    /// Variable name to use for entity bindings
    pub entity_var: Option<String>,
    /// Variable name to use for payload bindings
    pub payload_var: Option<String>,
    /// Variable name to use for user bindings
    pub user_var: Option<String>,
    /// Track local variable names
    pub locals: HashMap<String, String>,
    /// Indentation level
    pub indent_level: usize,
}

impl CodegenContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_local(&self, name: &str, py_name: &str) -> Self {
        let mut ctx = self.clone();
        ctx.locals.insert(name.to_string(), py_name.to_string());
        ctx
    }

    pub fn indent(&self) -> String {
        "    ".repeat(self.indent_level)
    }
}

/// Compile an OirExpr to Python code
pub fn compile_expr_to_py(expr: &OirExpr, ctx: &CodegenContext) -> String {
    match expr {
        OirExpr::Null => "None".to_string(),
        OirExpr::Bool { value } => if *value { "True" } else { "False" }.to_string(),
        OirExpr::Number { value } => {
            if value.fract() == 0.0 && value.abs() < 1e15 {
                format!("{}", *value as i64)
            } else {
                value.to_string()
            }
        }
        OirExpr::String { value } => format!("\"{}\"", escape_python_string(value)),
        OirExpr::Array { items } => {
            let compiled: Vec<String> = items.iter().map(|e| compile_expr_to_py(e, ctx)).collect();
            format!("[{}]", compiled.join(", "))
        }
        OirExpr::Object { fields } => compile_dict_literal(fields, ctx),
        OirExpr::Lambda { params, body } => {
            let new_ctx = params.iter().fold(ctx.clone(), |c, p| c.with_local(p, p));
            let body_code = compile_expr_to_py(body, &new_ctx);
            if params.is_empty() {
                format!("(lambda: {})", body_code)
            } else {
                format!("(lambda {}: {})", params.join(", "), body_code)
            }
        }
        OirExpr::Binding { binding } => compile_binding(binding, ctx),
        OirExpr::BinaryOp { op, left, right } => compile_binary_op(op, left, right, ctx),
        OirExpr::UnaryOp { op, operand } => compile_unary_op(op, operand, ctx),
        OirExpr::Conditional {
            condition,
            then_branch,
            else_branch,
        } => {
            let cond = compile_expr_to_py(condition, ctx);
            let then_expr = compile_expr_to_py(then_branch, ctx);
            let else_expr = compile_expr_to_py(else_branch, ctx);
            format!("({} if {} else {})", then_expr, cond, else_expr)
        }
        OirExpr::Call { function, args } => compile_function_call(function, args, ctx),
        OirExpr::Map {
            source,
            variable,
            body,
        } => {
            let src = compile_expr_to_py(source, ctx);
            let new_ctx = ctx.with_local(variable, variable);
            let body_code = compile_expr_to_py(body, &new_ctx);
            format!("[{} for {} in {}]", body_code, variable, src)
        }
        OirExpr::Filter {
            source,
            variable,
            predicate,
        } => {
            let src = compile_expr_to_py(source, ctx);
            let new_ctx = ctx.with_local(variable, variable);
            let pred = compile_expr_to_py(predicate, &new_ctx);
            format!("[{} for {} in {} if {}]", variable, variable, src, pred)
        }
        OirExpr::Reduce {
            source,
            variable,
            accumulator,
            initial,
            body,
        } => {
            let src = compile_expr_to_py(source, ctx);
            let init = compile_expr_to_py(initial, ctx);
            let new_ctx = ctx
                .with_local(variable, variable)
                .with_local(accumulator, accumulator);
            let body_code = compile_expr_to_py(body, &new_ctx);
            format!(
                "functools.reduce(lambda {}, {}: {}, {}, {})",
                accumulator, variable, body_code, src, init
            )
        }
        OirExpr::Effect { effect } => compile_effect(effect, ctx),
    }
}

/// Compile a dict literal
fn compile_dict_literal(fields: &orbital_core::ir::OirFields, ctx: &CodegenContext) -> String {
    if fields.is_empty() {
        return "{}".to_string();
    }

    let pairs: Vec<String> = fields
        .iter()
        .map(|(k, v)| {
            format!(
                "\"{}\": {}",
                escape_python_string(k),
                compile_expr_to_py(v, ctx)
            )
        })
        .collect();

    format!("{{ {} }}", pairs.join(", "))
}

/// Compile a binding reference (public alias used by trait_generator)
pub fn compile_binding_to_py(binding: &OirBinding, ctx: &CodegenContext) -> String {
    compile_binding(binding, ctx)
}

/// Compile a binding reference
fn compile_binding(binding: &OirBinding, ctx: &CodegenContext) -> String {
    let root = match &binding.root {
        OirBindingRoot::Entity => ctx.entity_var.as_deref().unwrap_or("entity"),
        OirBindingRoot::Payload => ctx.payload_var.as_deref().unwrap_or("payload"),
        OirBindingRoot::State => "state",
        OirBindingRoot::Config => "config",
        OirBindingRoot::User => ctx.user_var.as_deref().unwrap_or("user"),
        OirBindingRoot::Event
        | OirBindingRoot::PrevEvents
        | OirBindingRoot::PrevStates
        | OirBindingRoot::FromState
        | OirBindingRoot::ToState => {
            // The Python shell has no dispatch scope yet (G-ORB-SHELL-PY-002).
            return "None".to_string();
        }
        OirBindingRoot::Trait(_)
        | OirBindingRoot::Pages
        | OirBindingRoot::CurrentTheme
        | OirBindingRoot::Locale => {
            // `@trait.X` is a render-time client binding (resolved by
            // `<TraitFrame>` in `@almadar/ui`). It never applies to the
            // Python shell, which generates server-side logic. Emit `None`
            // so any accidental reference degrades cleanly instead of
            // producing invalid Python.
            return "None".to_string();
        }
        OirBindingRoot::Local(name) => {
            if let Some(py_name) = ctx.locals.get(name) {
                return py_name.clone();
            }
            name.as_str()
        }
    };

    if binding.path.is_empty() {
        root.to_string()
    } else {
        // Special handling for architecture field
        if matches!(&binding.root, OirBindingRoot::Entity)
            && binding.path.last().map(|s| s.as_str()) == Some("architecture")
        {
            return format!("self._get_architecture({}[\"id\"])", root);
        }

        // All roots use dict access (entity, payload, config are all dicts in Python)
        let path_access: Vec<String> = binding
            .path
            .iter()
            .map(|p| format!("[\"{}\"]", p))
            .collect();
        format!("{}{}", root, path_access.join(""))
    }
}

/// Compile a binary operation
fn compile_binary_op(
    op: &OirBinaryOp,
    left: &OirExpr,
    right: &OirExpr,
    ctx: &CodegenContext,
) -> String {
    let l = compile_expr_to_py(left, ctx);
    let r = compile_expr_to_py(right, ctx);

    match op {
        OirBinaryOp::Add => format!("({} + {})", l, r),
        OirBinaryOp::Sub => format!("({} - {})", l, r),
        OirBinaryOp::Mul => format!("({} * {})", l, r),
        OirBinaryOp::Div => format!("({} / {})", l, r),
        OirBinaryOp::Mod => format!("({} % {})", l, r),
        OirBinaryOp::Eq => format!("({} == {})", l, r),
        OirBinaryOp::Ne => format!("({} != {})", l, r),
        OirBinaryOp::Lt => format!("({} < {})", l, r),
        OirBinaryOp::Le => format!("({} <= {})", l, r),
        OirBinaryOp::Gt => format!("({} > {})", l, r),
        OirBinaryOp::Ge => format!("({} >= {})", l, r),
        OirBinaryOp::And => format!("({} and {})", l, r),
        OirBinaryOp::Or => format!("({} or {})", l, r),
        OirBinaryOp::Concat => format!("(str({}) + str({}))", l, r),
    }
}

/// Compile a unary operation
fn compile_unary_op(op: &OirUnaryOp, operand: &OirExpr, ctx: &CodegenContext) -> String {
    let val = compile_expr_to_py(operand, ctx);

    match op {
        OirUnaryOp::Not => format!("(not {})", val),
        OirUnaryOp::Neg => format!("(-{})", val),
    }
}

/// Compile a function call
fn compile_function_call(function: &str, args: &[OirExpr], ctx: &CodegenContext) -> String {
    // `let` in expression position (e.g. inside a `set` value or a lambda
    // predicate). Mirrors the TS IIFE (`(() => { const …; return body; })()`):
    // sequential walrus bindings in a tuple, each in scope for the next,
    // body as the final indexed element. (Lambda parameters would evaluate
    // all args up front and break the sequential dependency.)
    if function == "let" && args.len() == 2 {
        if let OirExpr::Array { items } = &args[0] {
            let mut local_ctx = ctx.clone();
            let mut parts: Vec<String> = Vec::new();
            for pair in items {
                if let OirExpr::Array { items: kv } = pair {
                    if kv.len() == 2 {
                        if let OirExpr::String { value: name } = &kv[0] {
                            let val = compile_expr_to_py(&kv[1], &local_ctx);
                            parts.push(format!("({name} := {val})"));
                            local_ctx = local_ctx.with_local(name, name);
                        }
                    }
                }
            }
            let n = parts.len();
            parts.push(compile_expr_to_py(&args[1], &local_ctx));
            if n == 0 {
                return parts
                    .into_iter()
                    .next()
                    .unwrap_or_else(|| "None".to_string());
            }
            return format!("({})[{}]", parts.join(", "), n);
        }
    }
    // `do` in expression position: value of the last sub-expression.
    // (Statement-position `do` is handled by `compile_effect_stmt`.)
    if function == "do" {
        if args.is_empty() {
            return "None".to_string();
        }
        let parts: Vec<String> = args.iter().map(|a| compile_expr_to_py(a, ctx)).collect();
        return format!("({},)[-1]", parts.join(", "));
    }

    let compiled_args: Vec<String> = args.iter().map(|a| compile_expr_to_py(a, ctx)).collect();

    match function {
        // Array operations
        "array/length" | "array/len" | "len" | "count" => {
            if compiled_args.is_empty() {
                "0".to_string()
            } else {
                format!("len({})", compiled_args[0])
            }
        }
        "append" | "array/concat" => {
            if compiled_args.len() >= 2 {
                format!("{} + [{}]", compiled_args[0], compiled_args[1])
            } else {
                "[]".to_string()
            }
        }
        "array/get" => {
            if compiled_args.len() >= 2 {
                format!("{}[{}]", compiled_args[0], compiled_args[1])
            } else {
                "None".to_string()
            }
        }
        "array/append" => {
            if compiled_args.len() >= 2 {
                format!("{} + [{}]", compiled_args[0], compiled_args[1])
            } else {
                "[]".to_string()
            }
        }
        "list" | "array/of" => format!("[{}]", compiled_args.join(", ")),
        "first" | "array/first" => {
            if !compiled_args.is_empty() {
                format!("({0}[0] if {0} else None)", compiled_args[0])
            } else {
                "None".to_string()
            }
        }
        "array/last" => {
            if !compiled_args.is_empty() {
                format!("({0}[-1] if {0} else None)", compiled_args[0])
            } else {
                "None".to_string()
            }
        }
        "array/nth" => {
            if compiled_args.len() >= 2 {
                format!(
                    "({0}[{1}] if -len({0}) <= {1} < len({0}) else None)",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "None".to_string()
            }
        }
        "array/drop" => {
            if compiled_args.len() >= 2 {
                format!("({0}[{1}:])", compiled_args[0], compiled_args[1])
            } else {
                compiled_args
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "[]".to_string())
            }
        }
        "array/dropLast" => {
            if compiled_args.len() >= 2 {
                format!(
                    "({0}[:len({0})-({1}) if len({0})>=({1}) else 0])",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                compiled_args
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "[]".to_string())
            }
        }
        "array/prepend" => {
            if compiled_args.len() >= 2 {
                format!("([{}] + {1})", compiled_args[1], compiled_args[0])
            } else {
                "[]".to_string()
            }
        }
        "array/includes" | "includes" => {
            if compiled_args.len() >= 2 {
                format!("({1} in {0})", compiled_args[0], compiled_args[1])
            } else {
                "False".to_string()
            }
        }
        "array/find" => {
            if compiled_args.len() >= 2 {
                format!(
                    "next((x for x in {0} if {1}), None)",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "None".to_string()
            }
        }
        "array/nearest" => {
            if compiled_args.len() >= 2 {
                format!("array_nearest({}, {})", compiled_args[0], compiled_args[1])
            } else {
                "None".to_string()
            }
        }
        "object/get" => {
            if compiled_args.len() >= 3 {
                format!(
                    "object_get({}, {}, {})",
                    compiled_args[0], compiled_args[1], compiled_args[2]
                )
            } else if compiled_args.len() == 2 {
                format!("object_get({}, {})", compiled_args[0], compiled_args[1])
            } else {
                "None".to_string()
            }
        }
        "object/keys" => format!(
            "list({}.keys())",
            compiled_args.first().map(|s| s.as_str()).unwrap_or("{}")
        ),
        "object/values" => format!(
            "list({}.values())",
            compiled_args.first().map(|s| s.as_str()).unwrap_or("{}")
        ),
        "object/merge" | "merge" => format!("object_merge({})", compiled_args.join(", ")),
        "object/omit" => {
            if compiled_args.len() >= 2 {
                format!(
                    "{{k: v for k, v in ({}).items() if k not in ({})}})",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                compiled_args
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "{}".to_string())
            }
        }
        "object/set" => {
            if compiled_args.len() >= 3 {
                format!(
                    "object_set({}, {}, {})",
                    compiled_args[0], compiled_args[1], compiled_args[2]
                )
            } else {
                compiled_args
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "{}".to_string())
            }
        }
        "object/clone" => format!(
            "dict({})",
            compiled_args.first().map(|s| s.as_str()).unwrap_or("{}")
        ),

        // Array higher-order (mirror TS operator.rs). Some survive as Call
        // form rather than OirExpr::Map/Filter, so handle them here too.
        "array/map" => {
            if compiled_args.len() >= 2 {
                format!("list(map({}, {}))", compiled_args[1], compiled_args[0])
            } else {
                "[]".to_string()
            }
        }
        "array/filter" => {
            if compiled_args.len() >= 2 {
                format!("array_filter({}, {})", compiled_args[0], compiled_args[1])
            } else {
                compiled_args
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "[]".to_string())
            }
        }
        "array/some" => {
            if compiled_args.len() >= 2 {
                format!(
                    "any({}(x) for x in ({}))",
                    compiled_args[1], compiled_args[0]
                )
            } else {
                "False".to_string()
            }
        }
        "array/every" => {
            if compiled_args.len() >= 2 {
                format!(
                    "all({}(x) for x in ({}))",
                    compiled_args[1], compiled_args[0]
                )
            } else {
                "True".to_string()
            }
        }
        "array/flatten" => {
            let depth = compiled_args.get(1).map(|s| s.as_str()).unwrap_or("1");
            if !compiled_args.is_empty() {
                format!("array_flatten({}, {})", compiled_args[0], depth)
            } else {
                "[]".to_string()
            }
        }
        "array/range" => {
            if compiled_args.len() >= 3 {
                format!(
                    "list(range({}, {}, {}))",
                    compiled_args[0], compiled_args[1], compiled_args[2]
                )
            } else if compiled_args.len() == 2 {
                format!("list(range({}, {}))", compiled_args[0], compiled_args[1])
            } else {
                "[]".to_string()
            }
        }
        "array/reduce" => {
            if compiled_args.len() >= 3 {
                format!(
                    "functools.reduce({}, {}, {})",
                    compiled_args[2], compiled_args[0], compiled_args[1]
                )
            } else {
                compiled_args
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "None".to_string())
            }
        }
        "array/sum" => format!(
            "sum({})",
            compiled_args.first().map(|s| s.as_str()).unwrap_or("[]")
        ),
        "array/avg" | "array/average" | "average" => {
            if !compiled_args.is_empty() {
                format!("(sum({0})/len({0}) if {0} else 0)", compiled_args[0])
            } else {
                "0".to_string()
            }
        }
        "array/repeat" => {
            if compiled_args.len() >= 2 {
                format!("[{}] * int({})", compiled_args[0], compiled_args[1])
            } else {
                "[]".to_string()
            }
        }
        "array/slice" => {
            if compiled_args.len() >= 3 {
                format!(
                    "({0}[{1}:{2}])",
                    compiled_args[0], compiled_args[1], compiled_args[2]
                )
            } else if compiled_args.len() == 2 {
                format!("({0}[{1}:])", compiled_args[0], compiled_args[1])
            } else {
                compiled_args
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "[]".to_string())
            }
        }
        "array/reverse" => format!(
            "({})[::-1]",
            compiled_args.first().map(|s| s.as_str()).unwrap_or("[]")
        ),
        "array/sort" => format!(
            "sorted({})",
            compiled_args.first().map(|s| s.as_str()).unwrap_or("[]")
        ),

        // math/random + grid/* (cells are {x,y} dicts — mirror TS operator.rs)
        "math/random" => "random.random()".to_string(),
        "math/mod" => {
            if compiled_args.len() >= 2 {
                format!("({0} % {1})", compiled_args[0], compiled_args[1])
            } else {
                "0".to_string()
            }
        }
        "math/randomInt" => {
            if compiled_args.len() >= 2 {
                format!("random.randint({}, {})", compiled_args[0], compiled_args[1])
            } else {
                "0".to_string()
            }
        }
        "grid/manhattan-distance" => {
            if compiled_args.len() >= 2 {
                format!("grid_manhattan({}, {})", compiled_args[0], compiled_args[1])
            } else {
                "0".to_string()
            }
        }
        "grid/distance" => {
            if compiled_args.len() >= 2 {
                format!("grid_chebyshev({}, {})", compiled_args[0], compiled_args[1])
            } else {
                "0".to_string()
            }
        }
        "grid/in-bounds" => {
            if compiled_args.len() >= 3 {
                format!(
                    "grid_in_bounds({}, {}, {})",
                    compiled_args[0], compiled_args[1], compiled_args[2]
                )
            } else {
                "False".to_string()
            }
        }
        "grid/to-world" => {
            if compiled_args.len() >= 2 {
                format!("{{\"x\": ({0}.get(\"x\",0) + 0.5) * {1}, \"y\": ({0}.get(\"y\",0) + 0.5) * {1}}}", compiled_args[0], compiled_args[1])
            } else {
                "{}".to_string()
            }
        }
        "grid/from-world" => {
            if compiled_args.len() >= 2 {
                format!(
                    "{{\"x\": int({0}.get(\"x\",0) // {1}), \"y\": int({0}.get(\"y\",0) // {1})}}",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "{}".to_string()
            }
        }

        // Math operations
        "math/abs" => format!("abs({})", compiled_args.join(", ")),
        "math/min" => format!("min({})", compiled_args.join(", ")),
        "math/max" => format!("max({})", compiled_args.join(", ")),
        "math/floor" => format!("math.floor({})", compiled_args.join(", ")),
        "math/ceil" => format!("math.ceil({})", compiled_args.join(", ")),
        "math/round" => format!("round({})", compiled_args.join(", ")),
        "math/sqrt" => format!("math.sqrt({})", compiled_args.join(", ")),
        "math/pow" => format!("math.pow({})", compiled_args.join(", ")),
        "math/log10" => format!("math.log10({})", compiled_args.join(", ")),
        "math/ln" => format!("math.log({})", compiled_args.join(", ")),
        "math/clamp" => {
            if compiled_args.len() >= 3 {
                format!(
                    "max({}, min({}, {}))",
                    compiled_args[1], compiled_args[2], compiled_args[0]
                )
            } else {
                "0".to_string()
            }
        }

        // Neural Network operations (PyTorch)
        // Both nn/ prefixed (legacy) and prefix-less (new) names supported
        "nn/sequential" | "sequential" => {
            format!("nn.Sequential(\n    {}\n)", compiled_args.join(",\n    "))
        }
        "nn/linear" | "linear" => {
            if compiled_args.len() >= 2 {
                format!("nn.Linear({}, {})", compiled_args[0], compiled_args[1])
            } else {
                "nn.Linear(1, 1)".to_string()
            }
        }
        "nn/relu" | "relu" => "nn.ReLU()".to_string(),
        "nn/tanh" | "tanh" => "nn.Tanh()".to_string(),
        "nn/sigmoid" | "sigmoid" => "nn.Sigmoid()".to_string(),
        "nn/softmax" | "softmax" => {
            let dim_arg = compiled_args.first().map(|s| s.as_str()).unwrap_or("-1");
            // Handle dict-style params: {"dim": -1} -> extract dim value
            if dim_arg.starts_with('{') {
                // Parse dim from dict: { "dim": -1 } -> -1
                if let Some(pos) = dim_arg.find("\"dim\":") {
                    let after = &dim_arg[pos + 6..];
                    let val = after
                        .trim()
                        .trim_start_matches(':')
                        .trim()
                        .trim_end_matches('}')
                        .trim()
                        .trim_end_matches(',')
                        .trim();
                    format!("nn.Softmax(dim={})", val)
                } else {
                    "nn.Softmax(dim=-1)".to_string()
                }
            } else {
                format!("nn.Softmax(dim={})", dim_arg)
            }
        }
        "nn/dropout" | "dropout" => {
            let p = compiled_args.first().map(|s| s.as_str()).unwrap_or("0.5");
            format!("nn.Dropout(p={})", p)
        }
        // Additional prefix-less activations
        "gelu" => "nn.GELU()".to_string(),
        "silu" => "nn.SiLU()".to_string(),
        "leaky-relu" => {
            let slope = compiled_args.first().map(|s| s.as_str()).unwrap_or("0.01");
            format!("nn.LeakyReLU(negative_slope={})", slope)
        }
        "elu" => {
            let alpha = compiled_args.first().map(|s| s.as_str()).unwrap_or("1.0");
            format!("nn.ELU(alpha={})", alpha)
        }
        "mish" => "nn.Mish()".to_string(),
        // Prefix-less norms
        "nn/batchnorm" | "batch-norm" => {
            if !compiled_args.is_empty() {
                format!("nn.BatchNorm1d({})", compiled_args[0])
            } else {
                "nn.BatchNorm1d(1)".to_string()
            }
        }
        "nn/layernorm" | "layer-norm" => {
            if !compiled_args.is_empty() {
                format!("nn.LayerNorm({})", compiled_args[0])
            } else {
                "nn.LayerNorm(1)".to_string()
            }
        }
        "group-norm" => {
            if compiled_args.len() >= 2 {
                format!("nn.GroupNorm({}, {})", compiled_args[0], compiled_args[1])
            } else {
                "nn.GroupNorm(1, 1)".to_string()
            }
        }
        "rms-norm" => {
            if !compiled_args.is_empty() {
                format!("RMSNorm({})", compiled_args[0])
            } else {
                "RMSNorm(1)".to_string()
            }
        }
        // Prefix-less pooling
        "max-pool2d" => {
            if !compiled_args.is_empty() {
                format!("nn.MaxPool2d({})", compiled_args[0])
            } else {
                "nn.MaxPool2d(2)".to_string()
            }
        }
        "avg-pool2d" => {
            if !compiled_args.is_empty() {
                format!("nn.AvgPool2d({})", compiled_args[0])
            } else {
                "nn.AvgPool2d(2)".to_string()
            }
        }
        "adaptive-avg-pool" => {
            if !compiled_args.is_empty() {
                format!("nn.AdaptiveAvgPool2d({})", compiled_args[0])
            } else {
                "nn.AdaptiveAvgPool2d(1)".to_string()
            }
        }
        "flatten" => "nn.Flatten()".to_string(),
        // Prefix-less convolutions
        "conv1d" => {
            if compiled_args.len() >= 3 {
                format!(
                    "nn.Conv1d({}, {}, {})",
                    compiled_args[0], compiled_args[1], compiled_args[2]
                )
            } else {
                "nn.Conv1d(1, 1, 1)".to_string()
            }
        }
        "conv2d" => {
            if compiled_args.len() >= 3 {
                format!(
                    "nn.Conv2d({}, {}, {})",
                    compiled_args[0], compiled_args[1], compiled_args[2]
                )
            } else {
                "nn.Conv2d(1, 1, 1)".to_string()
            }
        }
        "conv-transpose2d" => {
            if compiled_args.len() >= 3 {
                format!(
                    "nn.ConvTranspose2d({}, {}, {})",
                    compiled_args[0], compiled_args[1], compiled_args[2]
                )
            } else {
                "nn.ConvTranspose2d(1, 1, 1)".to_string()
            }
        }
        // Prefix-less recurrent
        "lstm" => {
            if compiled_args.len() >= 2 {
                format!(
                    "nn.LSTM({}, {}, batch_first=True)",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "nn.LSTM(1, 1, batch_first=True)".to_string()
            }
        }
        "gru" => {
            if compiled_args.len() >= 2 {
                format!(
                    "nn.GRU({}, {}, batch_first=True)",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "nn.GRU(1, 1, batch_first=True)".to_string()
            }
        }
        // Embedding
        "embedding" => {
            if compiled_args.len() >= 2 {
                format!("nn.Embedding({}, {})", compiled_args[0], compiled_args[1])
            } else {
                "nn.Embedding(1, 1)".to_string()
            }
        }
        "nn/forward" => {
            if compiled_args.len() >= 2 {
                format!("{}({})", compiled_args[0], compiled_args[1])
            } else {
                "None".to_string()
            }
        }
        "nn/getWeights" => {
            if !compiled_args.is_empty() {
                format!(
                    "torch.cat([p.flatten() for p in {}.parameters()])",
                    compiled_args[0]
                )
            } else {
                "torch.tensor([])".to_string()
            }
        }
        "nn/paramCount" => {
            if !compiled_args.is_empty() {
                format!("sum(p.numel() for p in {}.parameters())", compiled_args[0])
            } else {
                "0".to_string()
            }
        }
        "nn/setWeights" => {
            if compiled_args.len() >= 2 {
                format!("set_weights({}, {})", compiled_args[0], compiled_args[1])
            } else {
                "None".to_string()
            }
        }
        "nn/clone" => {
            if !compiled_args.is_empty() {
                format!("clone_module({})", compiled_args[0])
            } else {
                "None".to_string()
            }
        }

        // Tensor operations
        "tensor/from" => {
            if !compiled_args.is_empty() {
                format!("torch.tensor({}, dtype=torch.float32)", compiled_args[0])
            } else {
                "torch.tensor([])".to_string()
            }
        }
        "tensor/zeros" => format!("torch.zeros({})", compiled_args.join(", ")),
        "tensor/ones" => format!("torch.ones({})", compiled_args.join(", ")),
        "tensor/rand" => format!("torch.rand({})", compiled_args.join(", ")),
        "tensor/randn" => format!("torch.randn({})", compiled_args.join(", ")),
        "tensor/shape" => {
            if !compiled_args.is_empty() {
                format!("list({}.shape)", compiled_args[0])
            } else {
                "[]".to_string()
            }
        }
        "tensor/get" => {
            if compiled_args.len() >= 2 {
                format!("{}[{}].item()", compiled_args[0], compiled_args[1])
            } else {
                "0".to_string()
            }
        }
        "tensor/sum" => {
            if compiled_args.len() > 1 {
                format!("{}.sum(dim={})", compiled_args[0], compiled_args[1])
            } else if !compiled_args.is_empty() {
                format!("{}.sum()", compiled_args[0])
            } else {
                "0".to_string()
            }
        }
        "tensor/mean" => {
            if compiled_args.len() > 1 {
                format!("{}.mean(dim={})", compiled_args[0], compiled_args[1])
            } else if !compiled_args.is_empty() {
                format!("{}.mean()", compiled_args[0])
            } else {
                "0".to_string()
            }
        }
        "tensor/add" => {
            if compiled_args.len() >= 2 {
                format!("({} + {})", compiled_args[0], compiled_args[1])
            } else {
                "torch.tensor([])".to_string()
            }
        }
        "tensor/sub" => {
            if compiled_args.len() >= 2 {
                format!("({} - {})", compiled_args[0], compiled_args[1])
            } else {
                "torch.tensor([])".to_string()
            }
        }
        "tensor/mul" => {
            if compiled_args.len() >= 2 {
                format!("({} * {})", compiled_args[0], compiled_args[1])
            } else {
                "torch.tensor([])".to_string()
            }
        }
        "tensor/div" => {
            if compiled_args.len() >= 2 {
                format!("({} / {})", compiled_args[0], compiled_args[1])
            } else {
                "torch.tensor([])".to_string()
            }
        }
        "tensor/norm" => {
            if compiled_args.len() > 1 {
                format!("torch.norm({}, p={})", compiled_args[0], compiled_args[1])
            } else if !compiled_args.is_empty() {
                format!("torch.norm({})", compiled_args[0])
            } else {
                "0".to_string()
            }
        }
        "tensor/matmul" => {
            if compiled_args.len() >= 2 {
                format!("torch.matmul({}, {})", compiled_args[0], compiled_args[1])
            } else {
                "torch.tensor([])".to_string()
            }
        }
        "tensor/clamp" => {
            if compiled_args.len() >= 3 {
                format!(
                    "torch.clamp({}, {}, {})",
                    compiled_args[0], compiled_args[1], compiled_args[2]
                )
            } else {
                "torch.tensor([])".to_string()
            }
        }
        "tensor/clampPerDim" => {
            if compiled_args.len() >= 2 {
                format!("clamp_per_dim({}, {})", compiled_args[0], compiled_args[1])
            } else {
                "torch.tensor([])".to_string()
            }
        }
        "tensor/allInRange" => {
            if compiled_args.len() >= 2 {
                format!(
                    "(({} >= {}[0]).all() and ({} <= {}[1]).all())",
                    compiled_args[0], compiled_args[1], compiled_args[0], compiled_args[1]
                )
            } else {
                "True".to_string()
            }
        }
        "tensor/outOfRangeDims" => {
            if compiled_args.len() >= 2 {
                format!(
                    "out_of_range_dims({}, {})",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "[]".to_string()
            }
        }
        "tensor/toArray" | "tensor/toList" => {
            if !compiled_args.is_empty() {
                format!("{}.tolist()", compiled_args[0])
            } else {
                "[]".to_string()
            }
        }
        "tensor/cat" => {
            if compiled_args.len() >= 2 {
                format!(
                    "torch.cat([{}, {}], dim=-1)",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "torch.tensor([])".to_string()
            }
        }
        "tensor/stack" => {
            if compiled_args.len() >= 2 {
                format!("torch.stack([{}, {}])", compiled_args[0], compiled_args[1])
            } else {
                "torch.tensor([])".to_string()
            }
        }
        "tensor/transpose" => {
            if compiled_args.len() >= 3 {
                format!(
                    "{}.transpose({}, {})",
                    compiled_args[0], compiled_args[1], compiled_args[2]
                )
            } else if !compiled_args.is_empty() {
                format!("{}.t()", compiled_args[0])
            } else {
                "torch.tensor([])".to_string()
            }
        }
        "tensor/unsqueeze" => {
            if compiled_args.len() >= 2 {
                format!("{}.unsqueeze({})", compiled_args[0], compiled_args[1])
            } else {
                "torch.tensor([])".to_string()
            }
        }
        "tensor/squeeze" => {
            if !compiled_args.is_empty() {
                format!("{}.squeeze()", compiled_args[0])
            } else {
                "torch.tensor([])".to_string()
            }
        }
        "tensor/max" => {
            if compiled_args.len() > 1 {
                format!(
                    "torch.max({}, dim={}).values",
                    compiled_args[0], compiled_args[1]
                )
            } else if !compiled_args.is_empty() {
                format!("torch.max({})", compiled_args[0])
            } else {
                "0".to_string()
            }
        }
        "tensor/min" => {
            if compiled_args.len() > 1 {
                format!(
                    "torch.min({}, dim={}).values",
                    compiled_args[0], compiled_args[1]
                )
            } else if !compiled_args.is_empty() {
                format!("torch.min({})", compiled_args[0])
            } else {
                "0".to_string()
            }
        }
        "tensor/argmax" => {
            if compiled_args.len() > 1 {
                format!(
                    "torch.argmax({}, dim={})",
                    compiled_args[0], compiled_args[1]
                )
            } else if !compiled_args.is_empty() {
                format!("torch.argmax({})", compiled_args[0])
            } else {
                "0".to_string()
            }
        }
        "tensor/dot" => {
            if compiled_args.len() >= 2 {
                format!("torch.dot({}, {})", compiled_args[0], compiled_args[1])
            } else {
                "torch.tensor([])".to_string()
            }
        }
        "tensor/flatten" => {
            if !compiled_args.is_empty() {
                format!("torch.flatten({})", compiled_args[0])
            } else {
                "torch.tensor([])".to_string()
            }
        }
        "tensor/reshape" => {
            if compiled_args.len() >= 2 {
                format!("{}.reshape({})", compiled_args[0], compiled_args[1])
            } else if !compiled_args.is_empty() {
                compiled_args[0].clone()
            } else {
                "torch.tensor([])".to_string()
            }
        }
        "tensor/slice" => {
            if compiled_args.len() >= 3 {
                format!(
                    "{}[{}:{}]",
                    compiled_args[0], compiled_args[1], compiled_args[2]
                )
            } else if compiled_args.len() >= 2 {
                format!("{}[{}:]", compiled_args[0], compiled_args[1])
            } else {
                "torch.tensor([])".to_string()
            }
        }
        "tensor/outOfRangeIndices" => {
            if compiled_args.len() >= 2 {
                format!(
                    "out_of_range_indices({}, {})",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "[]".to_string()
            }
        }

        // Graph operations
        "graph/from-entities" => {
            format!("graph_from_entities({})", compiled_args.join(", "))
        }
        "graph/from-adjacency" => {
            if compiled_args.len() >= 2 {
                format!(
                    "graph_from_adjacency({}, {})",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "None".to_string()
            }
        }
        "graph/from-edge-list" => {
            if compiled_args.len() >= 2 {
                format!(
                    "graph_from_edge_list({}, {})",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "None".to_string()
            }
        }
        "graph/add-self-loops" => {
            if !compiled_args.is_empty() {
                format!("graph_add_self_loops({})", compiled_args[0])
            } else {
                "None".to_string()
            }
        }
        "graph/to-undirected" => {
            if !compiled_args.is_empty() {
                format!("graph_to_undirected({})", compiled_args[0])
            } else {
                "None".to_string()
            }
        }
        "graph/subgraph" => {
            if compiled_args.len() >= 2 {
                format!("graph_subgraph({}, {})", compiled_args[0], compiled_args[1])
            } else {
                "None".to_string()
            }
        }
        "graph/k-hop" => {
            if compiled_args.len() >= 3 {
                format!(
                    "graph_k_hop({}, {}, {})",
                    compiled_args[0], compiled_args[1], compiled_args[2]
                )
            } else {
                "None".to_string()
            }
        }
        "graph/node-features" => {
            if !compiled_args.is_empty() {
                format!("graph_node_features({})", compiled_args[0])
            } else {
                "None".to_string()
            }
        }
        "graph/edge-index" => {
            if !compiled_args.is_empty() {
                format!("graph_edge_index({})", compiled_args[0])
            } else {
                "None".to_string()
            }
        }
        "graph/edge-features" => {
            if !compiled_args.is_empty() {
                format!("graph_edge_features({})", compiled_args[0])
            } else {
                "None".to_string()
            }
        }
        "graph/num-nodes" => {
            if !compiled_args.is_empty() {
                format!("graph_num_nodes({})", compiled_args[0])
            } else {
                "0".to_string()
            }
        }
        "graph/num-edges" => {
            if !compiled_args.is_empty() {
                format!("graph_num_edges({})", compiled_args[0])
            } else {
                "0".to_string()
            }
        }
        "graph/degree" => {
            if !compiled_args.is_empty() {
                format!("graph_degree({})", compiled_args[0])
            } else {
                "None".to_string()
            }
        }
        "graph/batch" => {
            if !compiled_args.is_empty() {
                format!("graph_batch({})", compiled_args[0])
            } else {
                "None".to_string()
            }
        }

        // Contract operations
        "contract/validate-input" => {
            if compiled_args.len() >= 2 {
                format!(
                    "contract_validate_input({}, {})",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "None".to_string()
            }
        }
        "contract/validate-output" => {
            if compiled_args.len() >= 2 {
                format!(
                    "contract_validate_output({}, {})",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "None".to_string()
            }
        }
        "contract/clamp-output" => {
            if compiled_args.len() >= 2 {
                format!(
                    "contract_clamp_output({}, {})",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "None".to_string()
            }
        }
        "contract/violations" => {
            if compiled_args.len() >= 2 {
                format!(
                    "contract_violations({}, {})",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "[]".to_string()
            }
        }
        "contract/entity-to-tensor" => {
            if compiled_args.len() >= 2 {
                format!(
                    "contract_entity_to_tensor({}, {})",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "torch.tensor([])".to_string()
            }
        }
        "contract/tensor-to-payload" => {
            if compiled_args.len() >= 2 {
                format!(
                    "contract_tensor_to_payload({}, {})",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "{}".to_string()
            }
        }

        // Data operations
        "data/dataset" => {
            if compiled_args.len() >= 2 {
                format!("data_dataset({}, {})", compiled_args[0], compiled_args[1])
            } else {
                "None".to_string()
            }
        }
        "data/dataloader" => {
            if compiled_args.len() >= 2 {
                format!(
                    "data_dataloader({}, {})",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "None".to_string()
            }
        }
        "data/split" => {
            if compiled_args.len() >= 2 {
                format!("data_split({}, {})", compiled_args[0], compiled_args[1])
            } else {
                "None".to_string()
            }
        }
        "data/normalize" => {
            if compiled_args.len() >= 2 {
                format!("data_normalize({}, {})", compiled_args[0], compiled_args[1])
            } else {
                "None".to_string()
            }
        }
        "data/augment" => {
            if compiled_args.len() >= 2 {
                format!("data_augment({}, {})", compiled_args[0], compiled_args[1])
            } else {
                "None".to_string()
            }
        }
        "data/tokenize" => {
            if compiled_args.len() >= 2 {
                format!("data_tokenize({}, {})", compiled_args[0], compiled_args[1])
            } else {
                "None".to_string()
            }
        }
        "data/pad" => {
            if compiled_args.len() >= 2 {
                format!("data_pad({}, {})", compiled_args[0], compiled_args[1])
            } else {
                "None".to_string()
            }
        }

        // ML effect operations
        "forward" => {
            if compiled_args.len() >= 2 {
                format!("ml_forward({}, {})", compiled_args[0], compiled_args[1])
            } else if !compiled_args.is_empty() {
                format!("ml_forward({}, {{}})", compiled_args[0])
            } else {
                "None".to_string()
            }
        }
        "train" => {
            if !compiled_args.is_empty() {
                format!("ml_train({})", compiled_args[0])
            } else {
                "None".to_string()
            }
        }
        "evaluate" => {
            if !compiled_args.is_empty() {
                format!("ml_evaluate({})", compiled_args[0])
            } else {
                "None".to_string()
            }
        }
        "checkpoint/save" => {
            if compiled_args.len() >= 2 {
                format!(
                    "checkpoint_save({}, {})",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "None".to_string()
            }
        }
        "checkpoint/load" => {
            if !compiled_args.is_empty() {
                format!("checkpoint_load({})", compiled_args[0])
            } else {
                "None".to_string()
            }
        }

        // Training operations
        "train/loop" => {
            if compiled_args.len() >= 3 {
                format!(
                    "train_loop({}, {}, {})",
                    compiled_args[0], compiled_args[1], compiled_args[2]
                )
            } else {
                "None".to_string()
            }
        }
        "train/validate" => {
            if compiled_args.len() >= 2 {
                format!("validate({}, {})", compiled_args[0], compiled_args[1])
            } else {
                "None".to_string()
            }
        }
        "train/checkConstraints" => {
            if compiled_args.len() >= 2 {
                format!(
                    "check_constraints({}, {})",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "None".to_string()
            }
        }
        "train/clipGradients" => {
            // clipGradients takes gradient parameters (or an iterable of tensors) and max_norm
            // If arg is a model/module, call .parameters(); otherwise use directly
            if compiled_args.len() >= 2 {
                format!(
                    "torch.nn.utils.clip_grad_norm_({0}.parameters() if hasattr({0}, 'parameters') else ({0}.values() if isinstance({0}, dict) else {0}), {1})",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "None".to_string()
            }
        }
        "train/mse" => {
            if compiled_args.len() >= 2 {
                format!("F.mse_loss({}, {})", compiled_args[0], compiled_args[1])
            } else {
                "0".to_string()
            }
        }
        "train/adam" => {
            if compiled_args.len() >= 2 {
                // Second arg may be a config dict {"lr": value} or direct lr value
                let lr_arg = &compiled_args[1];
                if lr_arg.starts_with("{") && lr_arg.contains("\"lr\"") {
                    // Extract lr from config dict: {"lr": value} -> value["lr"]
                    format!(
                        "torch.optim.Adam({}.parameters(), lr={}[\"lr\"])",
                        compiled_args[0], lr_arg
                    )
                } else {
                    format!(
                        "torch.optim.Adam({}.parameters(), lr={})",
                        compiled_args[0], lr_arg
                    )
                }
            } else {
                "None".to_string()
            }
        }
        "train/clipWeights" => {
            if compiled_args.len() >= 2 {
                format!("clip_weights({}, {})", compiled_args[0], compiled_args[1])
            } else {
                "None".to_string()
            }
        }
        "train/checkWeightMagnitude" => {
            if compiled_args.len() >= 2 {
                format!(
                    "check_weight_magnitude({}, {})",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "True".to_string()
            }
        }
        "train/checkForbiddenOutputs" => {
            if compiled_args.len() >= 2 {
                format!(
                    "check_forbidden_outputs({}, {})",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "True".to_string()
            }
        }
        "train/step" => {
            if compiled_args.len() >= 4 {
                format!(
                    "train_step({}, {}, {}, {})",
                    compiled_args[0], compiled_args[1], compiled_args[2], compiled_args[3]
                )
            } else if compiled_args.len() >= 3 {
                format!(
                    "train_step({}, {}, {}, {{}})",
                    compiled_args[0], compiled_args[1], compiled_args[2]
                )
            } else {
                "None".to_string()
            }
        }
        "train/crossEntropy" => {
            if compiled_args.len() >= 2 {
                format!(
                    "F.cross_entropy({}, {})",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "0".to_string()
            }
        }
        "train/huber" => {
            if compiled_args.len() >= 2 {
                format!("F.huber_loss({}, {})", compiled_args[0], compiled_args[1])
            } else {
                "0".to_string()
            }
        }
        "train/sgd" => {
            if compiled_args.len() >= 2 {
                let lr_arg = &compiled_args[1];
                if lr_arg.starts_with('{') && lr_arg.contains("\"lr\"") {
                    format!(
                        "torch.optim.SGD({}.parameters(), lr={}[\"lr\"])",
                        compiled_args[0], lr_arg
                    )
                } else {
                    format!(
                        "torch.optim.SGD({}.parameters(), lr={})",
                        compiled_args[0], lr_arg
                    )
                }
            } else {
                "None".to_string()
            }
        }
        "train/checkRegression" => {
            if compiled_args.len() >= 2 {
                format!(
                    "check_regression({}, {})",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "True".to_string()
            }
        }
        "train/getGradientNorm" => {
            if !compiled_args.is_empty() {
                format!("get_gradient_norm({})", compiled_args[0])
            } else {
                "0.0".to_string()
            }
        }
        "train/getMaxWeightMagnitude" => {
            if !compiled_args.is_empty() {
                format!("get_max_weight_magnitude({})", compiled_args[0])
            } else {
                "0.0".to_string()
            }
        }
        "train/sampleBatch" => {
            if compiled_args.len() >= 2 {
                format!("sample_batch({}, {})", compiled_args[0], compiled_args[1])
            } else {
                "[]".to_string()
            }
        }
        "train/computeReturns" => {
            if compiled_args.len() >= 2 {
                format!(
                    "compute_returns({}, {})",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "torch.tensor([])".to_string()
            }
        }
        "train/computeAdvantages" => {
            if compiled_args.len() >= 3 {
                format!(
                    "compute_advantages({}, {}, {})",
                    compiled_args[0], compiled_args[1], compiled_args[2]
                )
            } else if compiled_args.len() >= 2 {
                format!(
                    "compute_advantages({}, {}, {{}})",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "torch.tensor([])".to_string()
            }
        }

        "math/sign" => {
            if !compiled_args.is_empty() {
                format!("(({0} > 0) - ({0} < 0))", compiled_args[0])
            } else {
                "0".to_string()
            }
        }
        "math/lerp" => {
            if compiled_args.len() >= 3 {
                format!(
                    "({0} + ({1} - {0}) * {2})",
                    compiled_args[0], compiled_args[1], compiled_args[2]
                )
            } else {
                "0".to_string()
            }
        }
        "str/default" => {
            if compiled_args.len() >= 2 {
                format!(
                    "({0} if {0} is not None else {1})",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                compiled_args
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "None".to_string())
            }
        }
        "array/max" => {
            if compiled_args.len() >= 2 {
                format!("array_max({}, {})", compiled_args[0], compiled_args[1])
            } else if !compiled_args.is_empty() {
                format!("(max({0}) if {0} else 0)", compiled_args[0])
            } else {
                "0".to_string()
            }
        }
        "array/min" => {
            if compiled_args.len() >= 2 {
                format!("array_min({}, {})", compiled_args[0], compiled_args[1])
            } else if !compiled_args.is_empty() {
                format!("(min({0}) if {0} else 0)", compiled_args[0])
            } else {
                "0".to_string()
            }
        }
        "array/take" => {
            if compiled_args.len() >= 2 {
                format!("({0}[:{1}])", compiled_args[0], compiled_args[1])
            } else {
                compiled_args
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "[]".to_string())
            }
        }
        "array/takeLast" => {
            if compiled_args.len() >= 2 {
                format!("({0}[-int({1}):])", compiled_args[0], compiled_args[1])
            } else {
                compiled_args
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "[]".to_string())
            }
        }
        "array/group" | "array/groupBy" => {
            if compiled_args.len() >= 2 {
                format!("array_group({}, {})", compiled_args[0], compiled_args[1])
            } else {
                "{}".to_string()
            }
        }
        "array/findIndex" => {
            if compiled_args.len() >= 2 {
                format!(
                    "next((i for i, x in enumerate({0}) if {1}(x)), -1)",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "-1".to_string()
            }
        }
        "array/shuffle" => {
            if !compiled_args.is_empty() {
                format!("random.sample({0}, len({0}))", compiled_args[0])
            } else {
                "[]".to_string()
            }
        }
        "object/entries" => format!(
            "[[k, v] for k, v in ({{}} if {0} is None else {0}).items()]",
            compiled_args.first().map(|s| s.as_str()).unwrap_or("{}")
        ),

        // ===== str/* (mirror TS operator.rs string category) =====
        "str/lower" => format!(
            "{}.lower()",
            compiled_args.first().map(|s| s.as_str()).unwrap_or("\"\"")
        ),
        // G-CROSS-041: the encoded quote body decodes to the quoted data.
        "quote" => format!(
            "json.loads({})",
            compiled_args
                .first()
                .map(|s| s.as_str())
                .unwrap_or("\"null\"")
        ),
        "str/upper" => format!(
            "{}.upper()",
            compiled_args.first().map(|s| s.as_str()).unwrap_or("\"\"")
        ),
        "str/trim" => format!(
            "{}.strip()",
            compiled_args.first().map(|s| s.as_str()).unwrap_or("\"\"")
        ),
        "str/len" | "str/length" => format!(
            "len({})",
            compiled_args.first().map(|s| s.as_str()).unwrap_or("\"\"")
        ),
        "str/concat" => format!("(\"\".join([{}]))", compiled_args.join(", ")),
        "str/includes" => {
            if compiled_args.len() >= 2 {
                format!("({1} in {0})", compiled_args[0], compiled_args[1])
            } else {
                "False".to_string()
            }
        }
        "str/split" => {
            if compiled_args.len() >= 2 {
                format!("({0}.split({1}))", compiled_args[0], compiled_args[1])
            } else {
                compiled_args
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "[]".to_string())
            }
        }
        "str/replace" => {
            if compiled_args.len() >= 3 {
                format!(
                    "({0}.replace({1}, {2}))",
                    compiled_args[0], compiled_args[1], compiled_args[2]
                )
            } else {
                compiled_args
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "\"\"".to_string())
            }
        }
        "str/slice" => {
            if compiled_args.len() >= 3 {
                format!(
                    "({0}[{1}:{2}])",
                    compiled_args[0], compiled_args[1], compiled_args[2]
                )
            } else if compiled_args.len() == 2 {
                format!("({0}[{1}:])", compiled_args[0], compiled_args[1])
            } else {
                compiled_args
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "\"\"".to_string())
            }
        }
        "str/starts-with" | "str/startsWith" => {
            if compiled_args.len() >= 2 {
                format!("({0}.startswith({1}))", compiled_args[0], compiled_args[1])
            } else {
                "False".to_string()
            }
        }
        "str/ends-with" | "str/endsWith" => {
            if compiled_args.len() >= 2 {
                format!("({0}.endswith({1}))", compiled_args[0], compiled_args[1])
            } else {
                "False".to_string()
            }
        }
        "str/join" => {
            if compiled_args.len() >= 2 {
                format!(
                    "({1}).join(str(x) for x in ({0}))",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "\"\"".to_string()
            }
        }

        "array/unique" => format!(
            "list(dict.fromkeys({0}))",
            compiled_args.first().map(|s| s.as_str()).unwrap_or("[]")
        ),
        "object/fromEntries" => format!(
            "dict({0})",
            compiled_args.first().map(|s| s.as_str()).unwrap_or("[]")
        ),
        "object/pick" => {
            if compiled_args.len() >= 2 {
                format!(
                    "{{k: ({0}).get(k) for k in ({1}) if k in ({0})}}",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "{}".to_string()
            }
        }

        // ===== time/* (epoch-millis model, mirrors TS operator.rs) =====
        "time/now" => "int(time.time() * 1000)".to_string(),
        "time/today" => "datetime.date.today().isoformat()".to_string(),
        "time/parse" => {
            if !compiled_args.is_empty() {
                format!("time_parse({0})", compiled_args[0])
            } else {
                "0".to_string()
            }
        }
        "time/format" => {
            if compiled_args.len() >= 2 {
                format!("time_format({0}, {1})", compiled_args[0], compiled_args[1])
            } else if !compiled_args.is_empty() {
                format!("time_format({0})", compiled_args[0])
            } else {
                "\"\"".to_string()
            }
        }
        "time/add" => {
            if compiled_args.len() >= 3 {
                format!(
                    "({0} + {1} * time_unit_ms({2}))",
                    compiled_args[0], compiled_args[1], compiled_args[2]
                )
            } else {
                compiled_args
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "0".to_string())
            }
        }
        "time/subtract" => {
            if compiled_args.len() >= 3 {
                format!(
                    "({0} - {1} * time_unit_ms({2}))",
                    compiled_args[0], compiled_args[1], compiled_args[2]
                )
            } else {
                compiled_args
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "0".to_string())
            }
        }
        "time/diff" => {
            if compiled_args.len() >= 3 {
                format!(
                    "int(({0} - {1}) / (time_unit_ms({2}) or 1))",
                    compiled_args[0], compiled_args[1], compiled_args[2]
                )
            } else {
                "0".to_string()
            }
        }
        "time/isBefore" => format!(
            "({0} < {1})",
            compiled_args.get(0).map(|s| s.as_str()).unwrap_or("0"),
            compiled_args.get(1).map(|s| s.as_str()).unwrap_or("0")
        ),
        "time/isAfter" => format!(
            "({0} > {1})",
            compiled_args.get(0).map(|s| s.as_str()).unwrap_or("0"),
            compiled_args.get(1).map(|s| s.as_str()).unwrap_or("0")
        ),
        "time/isSame" => format!(
            "({0} == {1})",
            compiled_args.get(0).map(|s| s.as_str()).unwrap_or("0"),
            compiled_args.get(1).map(|s| s.as_str()).unwrap_or("0")
        ),
        "time/isBetween" => {
            if compiled_args.len() >= 3 {
                format!(
                    "({0} >= {1} and {0} <= {2})",
                    compiled_args[0], compiled_args[1], compiled_args[2]
                )
            } else {
                "False".to_string()
            }
        }
        "time/isPast" => format!(
            "({0} < int(time.time() * 1000))",
            compiled_args.first().map(|s| s.as_str()).unwrap_or("0")
        ),
        "time/isFuture" => format!(
            "({0} > int(time.time() * 1000))",
            compiled_args.first().map(|s| s.as_str()).unwrap_or("0")
        ),
        "time/year" => format!(
            "time_year({0})",
            compiled_args.first().map(|s| s.as_str()).unwrap_or("0")
        ),
        "time/month" => format!(
            "time_month({0})",
            compiled_args.first().map(|s| s.as_str()).unwrap_or("0")
        ),
        "time/day" => format!(
            "time_day({0})",
            compiled_args.first().map(|s| s.as_str()).unwrap_or("0")
        ),
        "time/hour" => format!(
            "time_hour({0})",
            compiled_args.first().map(|s| s.as_str()).unwrap_or("0")
        ),
        "time/minute" => format!(
            "time_minute({0})",
            compiled_args.first().map(|s| s.as_str()).unwrap_or("0")
        ),
        "time/second" => format!(
            "time_second({0})",
            compiled_args.first().map(|s| s.as_str()).unwrap_or("0")
        ),
        "time/weekday" => format!(
            "time_weekday({0})",
            compiled_args.first().map(|s| s.as_str()).unwrap_or("0")
        ),

        // ===== validate/* (type predicates) =====
        "validate/number" => {
            if !compiled_args.is_empty() {
                format!(
                    "isinstance({0}, (int, float)) and not isinstance({0}, bool)",
                    compiled_args[0]
                )
            } else {
                "False".to_string()
            }
        }
        "validate/string" => format!(
            "isinstance({0}, str)",
            compiled_args.first().map(|s| s.as_str()).unwrap_or("None")
        ),
        "validate/boolean" => format!(
            "isinstance({0}, bool)",
            compiled_args.first().map(|s| s.as_str()).unwrap_or("None")
        ),
        "validate/array" => format!(
            "isinstance({0}, list)",
            compiled_args.first().map(|s| s.as_str()).unwrap_or("None")
        ),
        "validate/object" => format!(
            "isinstance({0}, dict)",
            compiled_args.first().map(|s| s.as_str()).unwrap_or("None")
        ),

        // ===== prob/* — statistical rung (Phase 4). Semantics ported from
        // packages/almadar-evaluator/std/prob.ts (the runtime evaluator).
        // Distributions/statistics use stdlib `random` + a `prob_helpers.py`
        // module; `prob/sample` inlines a comprehension (mirrors lazy
        // re-evaluation). The rejection-sampling trio (posterior/infer/
        // condition) re-evaluates arbitrary ASTs N times — not statically
        // compilable; emit a visible marker so it fails loudly at codegen.
        "prob/seed" => {
            if !compiled_args.is_empty() {
                format!("random.seed(int({}))", compiled_args[0])
            } else {
                "None".to_string()
            }
        }
        "prob/flip" => {
            if !compiled_args.is_empty() {
                format!("(random.random() < {})", compiled_args[0])
            } else {
                "False".to_string()
            }
        }
        "prob/gaussian" => {
            if compiled_args.len() >= 2 {
                format!("random.gauss({}, {})", compiled_args[0], compiled_args[1])
            } else {
                "0.0".to_string()
            }
        }
        "prob/uniform" => {
            if compiled_args.len() >= 2 {
                format!("random.uniform({}, {})", compiled_args[0], compiled_args[1])
            } else {
                "0.0".to_string()
            }
        }
        "prob/beta" => {
            if compiled_args.len() >= 2 {
                format!(
                    "random.betavariate({}, {})",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "0.0".to_string()
            }
        }
        "prob/categorical" => {
            if compiled_args.len() >= 2 {
                format!(
                    "random.choices({}, weights={}, k=1)[0]",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "None".to_string()
            }
        }
        "prob/poisson" => {
            if !compiled_args.is_empty() {
                format!("prob_poisson({})", compiled_args[0])
            } else {
                "0".to_string()
            }
        }
        "prob/sample" => {
            // args[1] is re-evaluated per iteration (lazy). Inline the
            // compiled body in a comprehension — exact mirror of
            // evalProbSample's per-iteration re-evaluation.
            if compiled_args.len() >= 2 {
                format!(
                    "[{} for _ in range({})]",
                    compiled_args[1], compiled_args[0]
                )
            } else {
                "[]".to_string()
            }
        }
        "prob/expected-value" | "prob/mean" => {
            if !compiled_args.is_empty() {
                format!("prob_expected_value({})", compiled_args[0])
            } else {
                "0".to_string()
            }
        }
        "prob/variance" => {
            if !compiled_args.is_empty() {
                format!("prob_variance({})", compiled_args[0])
            } else {
                "0".to_string()
            }
        }
        "prob/percentile" => {
            if compiled_args.len() >= 2 {
                format!(
                    "prob_percentile({}, {})",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "0".to_string()
            }
        }
        "prob/credible-interval" => {
            if compiled_args.len() >= 2 {
                format!(
                    "prob_credible_interval({}, {})",
                    compiled_args[0], compiled_args[1]
                )
            } else {
                "[0, 0]".to_string()
            }
        }
        "prob/histogram" => {
            if compiled_args.len() >= 2 {
                format!("prob_histogram({}, {})", compiled_args[0], compiled_args[1])
            } else {
                "{\"binEdges\": [], \"counts\": []}".to_string()
            }
        }
        "prob/posterior" | "prob/infer" | "prob/condition" => {
            // Rejection sampling re-evaluates arbitrary model/evidence/query
            // ASTs N times against a fresh entity with a rejection flag
            // (evalProbPosterior/Infer/Condition in std/prob.ts). Compiled
            // Python has no AST to re-evaluate. Emit a visible marker so any
            // behavior using it fails loudly at codegen (run on the JS path).
            let fn_name = function.replace('/', "_").replace('-', "_");
            format!(
                "# {} — {}({})",
                orbital_core::unsupported_operator_marker(function),
                fn_name,
                compiled_args.join(", ")
            )
        }

        // Unknown operator — emit the shared marker so drift is caught by the
        // pre-write output scan (generate.rs) and the registry parity test.
        _ => {
            let fn_name = function.replace('/', "_").replace('-', "_");
            format!(
                "# {} — {}({})",
                orbital_core::unsupported_operator_marker(function),
                fn_name,
                compiled_args.join(", ")
            )
        }
    }
}

/// Compile an effect
fn compile_effect(effect: &OirEffect, ctx: &CodegenContext) -> String {
    match effect {
        OirEffect::Set { target, value, .. } => {
            let target_code = compile_binding(target, ctx);
            let value_code = compile_expr_to_py(value, ctx);

            // Special handling for architecture field
            if target.path.last().map(|s| s.as_str()) == Some("architecture") {
                // Use dummy assignment `_ = ` so trait generator treats it as a statement
                format!("_ = self._set_architecture(entity[\"id\"], {})", value_code)
            } else {
                format!("{} = {}", target_code, value_code)
            }
        }
        OirEffect::SetDynamic { target, value, .. } => {
            // Dynamic path set - target is an expression that evaluates to a path string
            let target_code = compile_expr_to_py(target, ctx);
            let value_code = compile_expr_to_py(value, ctx);
            format!("set_dynamic_path(entity, {}, {})", target_code, value_code)
        }
        OirEffect::Emit { event, payload } => {
            let payload_code = if let Some(p) = payload {
                compile_expr_to_py(p, ctx)
            } else {
                "{}".to_string()
            };
            format!("await event_bus.emit(\"{}\", {})", event, payload_code)
        }
        OirEffect::Log { level, message } => {
            let msg = compile_expr_to_py(message, ctx);
            format!("print(\"[{:?}]\", {})", level, msg)
        }
        OirEffect::Navigate { path, params, .. } => {
            let path_code = compile_expr_to_py(path, ctx);
            if let Some(p) = params {
                let params_code = compile_expr_to_py(p, ctx);
                format!("navigate({}, {})", path_code, params_code)
            } else {
                format!("navigate({})", path_code)
            }
        }
        OirEffect::NavigateBack => "navigate_back()".to_string(),
        OirEffect::Persist { .. } => {
            // Persist is handled at the transition level in generate_transition_effects.
            // When this arm fires it means the expression appears in a non-effect position
            // (e.g. inside a guard), which is unsupported — emit a no-op.
            "None".to_string()
        }
        OirEffect::RenderUi { slot, pattern } => {
            let pattern_code = compile_pattern(pattern);
            format!("render_ui(\"{}\", {})", slot, pattern_code)
        }
        OirEffect::Fetch {
            entity_type,
            id,
            filter,
            order_by,
            offset,
            limit,
            include: _,
            ..
        } => {
            if let Some(id_expr) = id {
                format!(
                    "await fetch_by_id(entity_type=\"{}\", id={})",
                    entity_type,
                    compile_expr_to_py(id_expr, ctx)
                )
            } else {
                let mut parts = vec![format!("entity_type=\"{}\"", entity_type)];
                if let Some(f) = filter {
                    parts.push(format!("filter={}", compile_expr_to_py(f, ctx)));
                }
                if let Some(o) = order_by {
                    parts.push(format!("order_by={}", compile_expr_to_py(o, ctx)));
                }
                if let Some(o) = offset {
                    parts.push(format!("offset={}", compile_expr_to_py(o, ctx)));
                }
                if let Some(l) = limit {
                    parts.push(format!("limit={}", compile_expr_to_py(l, ctx)));
                }
                format!("await fetch({})", parts.join(", "))
            }
        }
        OirEffect::CallService {
            service,
            method,
            args,
            ..
        } => {
            let args_code = compile_expr_to_py(args, ctx);
            format!(
                "await call_service(\"{}\", \"{}\", {})",
                service, method, args_code
            )
        }
        OirEffect::PersistBatch { .. } => {
            format!(
                "# {} ({})",
                orbital_core::unsupported_operator_marker("persist batch"),
                PUSHDOWN_STATUS
            )
        }
        OirEffect::CancelCall { .. } => {
            format!(
                "# {}",
                orbital_core::unsupported_operator_marker("cancel-call")
            )
        }
        OirEffect::Http { .. } => {
            format!("# {}", orbital_core::unsupported_operator_marker("http"))
        }
        OirEffect::Spawn { entity_type, props } => {
            let props_code = compile_expr_to_py(props, ctx);
            format!("spawn(\"{}\", {})", entity_type, props_code)
        }
        OirEffect::Despawn { entity_id } => {
            let id_code = compile_expr_to_py(entity_id, ctx);
            format!("despawn({})", id_code)
        }
        OirEffect::Custom { name, args, .. } => {
            let args_code: Vec<String> = args
                .iter()
                .map(|(k, v)| format!("{}={}", k, compile_expr_to_py(v, ctx)))
                .collect();
            format!("custom_effect_{}({})", name, args_code.join(", "))
        }
        // OS trigger effects — server-side only, generate comment in Python
        OirEffect::OsWatchFiles { glob, .. } => {
            format!(
                "# server-side: os/watch-files({})",
                compile_expr_to_py(glob, ctx)
            )
        }
        OirEffect::OsWatchProcess { name, .. } => {
            format!(
                "# server-side: os/watch-process({})",
                compile_expr_to_py(name, ctx)
            )
        }
        OirEffect::OsWatchPort { port, .. } => {
            format!(
                "# server-side: os/watch-port({})",
                compile_expr_to_py(port, ctx)
            )
        }
        OirEffect::OsWatchHttp { url_pattern, .. } => {
            format!(
                "# server-side: os/watch-http({})",
                compile_expr_to_py(url_pattern, ctx)
            )
        }
        OirEffect::OsWatchCron { expression, .. } => {
            format!(
                "# server-side: os/watch-cron({})",
                compile_expr_to_py(expression, ctx)
            )
        }
        OirEffect::OsWatchSignal { signal, .. } => {
            format!(
                "# server-side: os/watch-signal({})",
                compile_expr_to_py(signal, ctx)
            )
        }
        OirEffect::OsWatchEnv { variable, .. } => {
            format!(
                "# server-side: os/watch-env({})",
                compile_expr_to_py(variable, ctx)
            )
        }
        OirEffect::OsDebounce { ms, event_type } => {
            format!(
                "# server-side: os/debounce({}, {})",
                compile_expr_to_py(ms, ctx),
                compile_expr_to_py(event_type, ctx)
            )
        }
        OirEffect::Ref {
            entity_type, id, ..
        } => {
            if let Some(id_expr) = id {
                format!(
                    "await entity_store.ref_entity('{}', id={})",
                    entity_type,
                    compile_expr_to_py(id_expr, ctx)
                )
            } else {
                format!("await entity_store.ref_entity('{}')", entity_type)
            }
        }
        OirEffect::Deref { entity_type, id } => {
            if let Some(id_expr) = id {
                format!(
                    "await entity_store.deref('{}', {})",
                    entity_type,
                    compile_expr_to_py(id_expr, ctx)
                )
            } else {
                format!("entity_store.get_snapshot('{}')", entity_type)
            }
        }
        OirEffect::Swap {
            entity_type,
            id,
            transform,
        } => {
            format!(
                "await entity_store.swap('{}', {}, lambda current: {})",
                entity_type,
                compile_expr_to_py(id, ctx),
                compile_expr_to_py(transform, ctx)
            )
        }
        OirEffect::Watch { entity_type, .. } => {
            format!("# client-side: watch('{}', [effects])", entity_type)
        }
        OirEffect::Atomic { .. } => "await entity_store.run_transaction(effects)".to_string(),
        OirEffect::FetchStream { entity_type, .. } => {
            // fetch-stream is server-side streaming — Python backend emits a no-op stub
            format!(
                "# fetch-stream('{}') not supported in Python shell",
                entity_type
            )
        }
        OirEffect::SendServer { event, .. } => {
            // Client-only — Python shell emits a stub comment
            format!(
                "# send-server({:?}) client-only, not applicable in Python shell",
                event
            )
        }
        OirEffect::BrowserOpenFilePicker { options, .. } => {
            // Client device API — executed by the browser host, not the Python shell.
            match options {
                Some(o) => format!(
                    "# client-device: browser/open-file-picker({})",
                    compile_expr_to_py(o, ctx)
                ),
                None => "# client-device: browser/open-file-picker".to_string(),
            }
        }
        OirEffect::BrowserClipboardRead { .. } => {
            "# client-device: browser/clipboard-read".to_string()
        }
        OirEffect::BrowserPushSubscribe { .. } => {
            "# client-device: browser/push-subscribe".to_string()
        }
        OirEffect::BrowserClipboardWrite { text, .. } => {
            format!(
                "# client-device: browser/clipboard-write({})",
                compile_expr_to_py(text, ctx)
            )
        }
        OirEffect::BrowserGeolocationCurrent { options, .. } => match options {
            Some(o) => format!(
                "# client-device: browser/geolocation-current({})",
                compile_expr_to_py(o, ctx)
            ),
            None => "# client-device: browser/geolocation-current".to_string(),
        },
    }
}

/// Compile a pattern to Python
fn compile_pattern(pattern: &OirPattern) -> String {
    let config_parts: Vec<String> = pattern
        .config
        .iter()
        .map(|(k, v)| format!("\"{}\": {}", k, compile_oir_value(v)))
        .collect();
    format!(
        "{{ \"pattern_type\": \"{}\", \"config\": {{ {} }} }}",
        pattern.pattern_type,
        config_parts.join(", ")
    )
}

/// Compile an OirValue to Python
fn compile_oir_value(value: &orbital_core::ir::OirValue) -> String {
    use orbital_core::ir::OirValue;
    if let Some(data) = orbital_core::schema::quote::quoted_oir_value(value) {
        return compile_oir_value(&data);
    }
    match value {
        OirValue::Null => "None".to_string(),
        OirValue::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        OirValue::Number(n) => {
            if n.fract() == 0.0 && n.abs() < 1e15 {
                format!("{}", *n as i64)
            } else {
                n.to_string()
            }
        }
        OirValue::String(s) => format!("\"{}\"", escape_python_string(s)),
        OirValue::Array(items) => {
            let compiled: Vec<String> = items.iter().map(|v| compile_oir_value(v)).collect();
            format!("[{}]", compiled.join(", "))
        }
        OirValue::Object(fields) => {
            let pairs: Vec<String> = fields
                .iter()
                .map(|(k, v)| format!("\"{}\": {}", k, compile_oir_value(v)))
                .collect();
            format!("{{ {} }}", pairs.join(", "))
        }
    }
}

/// Declared reason (owner Q13, G-ORBITAL-SHELL-PYTHON-001) the Python shell
/// stays on the plain data path: full listing then in-process filter/order,
/// no query push-down, no `writeWhere`, no `persist batch`. The in-process order of an unordered fetch is the
/// default one (`createdAt`, then `id`, EXEC-FETCH-1): `core/row_order.py`.
pub const PUSHDOWN_STATUS: &str = "python shell has no query/write push-down or persist batch: \
declared plain path, owner Q13, G-ORBITAL-SHELL-PYTHON-001";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_expressions() {
        let ctx = CodegenContext::default();

        assert_eq!(compile_expr_to_py(&OirExpr::Null, &ctx), "None");
        assert_eq!(
            compile_expr_to_py(&OirExpr::Bool { value: true }, &ctx),
            "True"
        );
        assert_eq!(
            compile_expr_to_py(&OirExpr::Number { value: 42.0 }, &ctx),
            "42"
        );
    }

    /// §48: a navigate carrying a payload must emit the resolved `params`
    /// into the Python `navigate(path, params)` call — `compile_effect`
    /// already branched on `params` (unlike the TS backend, which dropped
    /// it at resolve time); this pins that the resolved payload actually
    /// reaches the emitted call.
    #[test]
    fn navigate_with_payload_emits_params_in_python() {
        let ctx = CodegenContext::default();
        let mut fields = orbital_core::ir::OirFields::new();
        fields.insert(
            "id".to_string(),
            OirExpr::String {
                value: "@entity.id".to_string(),
            },
        );
        let effect = OirEffect::Navigate {
            path: OirExpr::String {
                value: "/detail".to_string(),
            },
            params: Some(OirExpr::Object { fields }),
            crumb: None,
        };
        let out = compile_effect(&effect, &ctx);
        assert!(
            out.starts_with("navigate(") && out.contains("\"id\""),
            "expected the resolved payload in the emitted navigate call, got: {}",
            out
        );
    }

    #[test]
    fn unknown_operator_emits_unsupported_marker() {
        let ctx = CodegenContext::default();
        let expr = OirExpr::Call {
            function: "my/unknown-op".to_string(),
            args: vec![OirExpr::Number { value: 1.0 }],
        };
        let out = compile_expr_to_py(&expr, &ctx);
        assert!(
            out.contains("# UNSUPPORTED operator: my/unknown-op"),
            "expected UNSUPPORTED marker in output, got: {}",
            out
        );
    }

    #[test]
    fn cancel_call_is_not_silently_dropped_by_the_python_shell() {
        let ctx = CodegenContext::default();
        let effect = OirEffect::cancel_call(OirExpr::string("batch-1"));
        let out = compile_effect(&effect, &ctx);
        assert!(
            out.contains("UNSUPPORTED operator: cancel-call"),
            "python has no call registry; the refusal must be loud, got: {out}"
        );
    }

    #[test]
    fn http_is_not_silently_dropped_by_the_python_shell() {
        let ctx = CodegenContext::default();
        let effect = OirEffect::http(
            orbital_core::runtime::http_request::HttpMethod::Get,
            OirExpr::string("https://api.example.com"),
            OirExpr::string("/x"),
        );
        let out = compile_effect(&effect, &ctx);
        assert!(
            out.contains("UNSUPPORTED operator: http"),
            "python has no egress-enforcing http client; the refusal must be loud, got: {out}"
        );
        assert!(!out.contains("api.example.com"), "{out}");
    }

    #[test]
    fn persist_batch_is_not_silently_dropped_by_the_python_shell() {
        let ctx = CodegenContext::default();
        let effect = OirEffect::persist_batch(OirExpr::array(vec![]), vec![]);
        let out = compile_effect(&effect, &ctx);
        assert!(
            out.contains("UNSUPPORTED operator: persist batch"),
            "python has no batch writer; the refusal must be loud, got: {out}"
        );
    }

    // Registry-driven operator-completeness gate — the Python analogue of the
    // Rust evaluator parity gate (orbital-compiler `embedded.rs`
    // `rust_evaluator_covers_every_canonical_rust_operator`). The Python
    // dispatch must have a real arm for EVERY operator whose canonical category
    // targets `python`. Operators are open strings, so `canonical-operators.json`
    // filtered by `target` is the closed set that stands in for the enum: any
    // python-target op that falls through to the UNSUPPORTED catch-all fails this
    // test with zero behaviors compiled.
    /// G-CROSS-041: a quoted pattern-config value is written as the data it holds.
    #[test]
    fn a_quoted_literal_compiles_to_its_data() {
        use orbital_core::ir::OirValue;
        let sexpr: orbital_core::schema::SExpression =
            serde_json::from_str(r#"[">", "@payload.amount", 0]"#).unwrap();
        let quoted = OirValue::Array(vec![
            OirValue::String("quote".into()),
            OirValue::String(orbital_core::schema::quote::encode_quote_body(&sexpr)),
        ]);
        assert_eq!(compile_oir_value(&quoted), r#"[">", "@payload.amount", 0]"#);
        let list = OirValue::Array(vec![
            OirValue::String("quotes".into()),
            OirValue::String("{}".into()),
        ]);
        assert_eq!(compile_oir_value(&list), r#"["quotes", "{}"]"#, "control");
    }

    /// G-CROSS-041: the compiled `quote` decodes each shared fixture body to the
    /// same data the evaluator returns (`fixtures/quote/cases.json`).
    #[test]
    fn compiled_quote_decodes_the_shared_fixture() {
        let ctx = CodegenContext::default();
        let cases: Vec<serde_json::Value> =
            serde_json::from_str(orbital_core::fixtures::quote_cases()).unwrap();
        for case in cases {
            let body = case["body"].as_str().unwrap().to_string();
            let py = compile_function_call("quote", &[OirExpr::String { value: body }], &ctx);
            let script = format!("import json, sys\nsys.stdout.write(json.dumps({py}))");
            let out = std::process::Command::new("python3")
                .args(["-c", &script])
                .output()
                .expect("python3 must be on PATH to run the emitted quote code");
            assert!(
                out.status.success(),
                "{}: {}",
                case["name"],
                String::from_utf8_lossy(&out.stderr)
            );
            let got: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
            assert_eq!(got, case["expr"], "{}", case["name"]);
        }
    }

    #[test]
    fn python_dispatch_covers_every_canonical_python_operator() {
        let ctx = CodegenContext::default();
        let mut missing = Vec::new();
        for name in orbital_core::language::registry::operators().keys() {
            if !orbital_core::language::registry::is_python_target(name) {
                continue;
            }
            let out = compile_function_call(name, &[], &ctx);
            if out.contains(orbital_core::UNSUPPORTED_OPERATOR_MARKER) {
                let cat = orbital_core::language::registry::get_operator(name)
                    .map(|o| o.category.as_str())
                    .unwrap_or("?");
                missing.push(format!("{name} ({cat})"));
            }
        }
        missing.sort();
        assert!(
            missing.is_empty(),
            "Python dispatch missing {} canonical python-target operator(s) \
             (fall through to UNSUPPORTED):\n  {}",
            missing.len(),
            missing.join("\n  ")
        );
    }

    #[test]
    fn pushdown_status_names_reason_and_gap() {
        assert!(PUSHDOWN_STATUS.contains("Q13"));
        assert!(PUSHDOWN_STATUS.contains("G-ORBITAL-SHELL-PYTHON-001"));
    }

    #[test]
    fn ordered_filtered_fetch_stays_on_plain_path() {
        let ctx = CodegenContext::default();
        let effect = OirEffect::Fetch {
            entity_type: "Task".to_string(),
            id: None,
            filter: Some(OirExpr::Bool { value: true }),
            order_by: Some(OirExpr::String {
                value: "title".to_string(),
            }),
            offset: None,
            limit: Some(OirExpr::Number { value: 5.0 }),
            include: None,
            emit: None,
        };
        let out = compile_effect(&effect, &ctx);
        assert!(out.starts_with("await fetch(entity_type=\"Task\""), "{out}");
        assert!(out.contains("filter=") && out.contains("order_by=") && out.contains("limit="));
        assert!(
            !out.contains("query") && !out.contains("write_where"),
            "{out}"
        );
    }

    #[test]
    fn persist_batch_fails_loudly_with_reason() {
        let ctx = CodegenContext::default();
        let effect = OirEffect::persist_batch(OirExpr::Null, Vec::new());
        let out = compile_effect(&effect, &ctx);
        assert!(
            out.contains("# UNSUPPORTED operator: persist batch"),
            "{out}"
        );
        assert!(out.contains(PUSHDOWN_STATUS), "{out}");
    }
}
