//! `orb-shell-python`: the Python shell as a program (`orbital_core::shell_protocol`).

use crate::kernel_host::KernelProject;
use crate::PythonBackend;
use orbital_core_base::generate::ShellMode;
use orbital_core_base::project::write_project;
use orbital_core_base::shell_protocol::{
    description, written_files, GenerateRequest, GenerateResponse, ShellDescription, ShellFailure,
    ShellProgram, CODE_BAD_REQUEST, CODE_GENERATE_FAILED,
};
use std::path::Path;

pub struct PythonProgram;

/// The native kernel extension, shipped beside the shell program and copied into the project.
const KERNEL_EXTENSION: &str = if cfg!(windows) {
    "orbital_kernel.pyd"
} else {
    "orbital_kernel.abi3.so"
};

fn kernel_project(request: &GenerateRequest) -> Result<KernelProject, ShellFailure> {
    let served = request.served.as_ref().ok_or_else(|| {
        ShellFailure::new(
            CODE_BAD_REQUEST,
            "the python shell serves a program on the native kernel, and the request carries no served program",
        )
    })?;
    let value = serde_json::to_value(served).map_err(|e| {
        ShellFailure::new(
            CODE_GENERATE_FAILED,
            format!("serialize the served program: {e}"),
        )
    })?;
    let program = serde_json::to_string(&orbital_core_eval::util::canonical_json(&value)).map_err(|e| {
        ShellFailure::new(
            CODE_GENERATE_FAILED,
            format!("serialize the served program: {e}"),
        )
    })?;
    let language = orbital_core_base::language_data::language_bundle_json()
        .map_err(|e| ShellFailure::new(CODE_GENERATE_FAILED, format!("{}: {e}", e.code())))?;
    Ok(KernelProject { program, language })
}

impl ShellProgram for PythonProgram {
    fn describe(&self) -> ShellDescription {
        let mut described = description("python", env!("CARGO_PKG_VERSION"), true, vec![]);
        described.needs_served = true;
        described
    }

    fn generate(
        &self,
        request: &GenerateRequest,
        output: &Path,
    ) -> Result<GenerateResponse, ShellFailure> {
        let serves = request.write.mode != ShellMode::Client;
        let mut backend = PythonBackend::new();
        if serves {
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
        if let (true, Some(source)) = (serves, &request.source_path) {
            orbital_core_base::project::copy_locales(source, output)
                .map_err(|e| ShellFailure::new(CODE_GENERATE_FAILED, format!("copy locales: {e}")))?;
        }
        if serves {
            warnings.extend(
                orbital_core_base::shell_protocol::ship_kernel_beside_program(
                    KERNEL_EXTENSION,
                    output,
                )?,
            );
        }
        Ok(GenerateResponse {
            files: written_files(&generated),
            warnings,
            next_steps: vec![
                format!("cd {}", output.display()),
                "python -m venv .venv && source .venv/bin/activate".to_string(),
                "pip install -e .".to_string(),
                "python -m orbital_app".to_string(),
            ],
        })
    }

    fn emit(
        &self,
        request: &GenerateRequest,
    ) -> Result<orbital_core_base::generate::GeneratedOutput, ShellFailure> {
        use orbital_core_base::generate::Backend as _;
        Ok(PythonBackend::new().generate(&request.module))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use orbital_core_base::ir::OirModule;
    use orbital_core_base::schema::types::OrbitalSchema;
    use orbital_core_base::shell_protocol::IR_VERSION;

    fn served(name: &str) -> OrbitalSchema {
        serde_json::from_value(serde_json::json!({
            "name": name,
            "orbitals": [{
                "name": name,
                "entity": {
                    "name": "E",
                    "persistence": "runtime",
                    "fields": [{ "name": "id", "type": "string" }]
                },
                "traits": [],
                "pages": []
            }]
        }))
        .unwrap()
    }

    fn request(mode: ShellMode, served: Option<OrbitalSchema>) -> GenerateRequest {
        GenerateRequest {
            ir_version: IR_VERSION,
            module: OirModule::new("app", "1.0.0"),
            resolved: None,
            served,
            source_path: None,
            write: orbital_core_base::project::WriteOptions {
                mode,
                ..Default::default()
            },
            options: Default::default(),
        }
    }

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("orb-py-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn the_shell_asks_for_the_resolved_and_the_served_program() {
        let description = PythonProgram.describe();
        assert!(description.needs_resolved);
        assert!(description.needs_served);
        assert!(description.options.is_empty());
    }

    #[test]
    fn a_full_generate_without_a_served_program_is_refused() {
        let out = scratch("full-none");
        let failure = PythonProgram
            .generate(&request(ShellMode::Full, None), &out)
            .unwrap_err();
        assert_eq!(failure.code, CODE_BAD_REQUEST);
        assert!(
            failure.message.contains("served program"),
            "{}",
            failure.message
        );
        let server = PythonProgram
            .generate(&request(ShellMode::Server, None), &scratch("server-none"))
            .unwrap_err();
        assert_eq!(server.code, CODE_BAD_REQUEST);
    }

    #[test]
    fn control_client_mode_is_not_refused_without_a_served_program() {
        let out = scratch("client-none");
        let result = PythonProgram.generate(&request(ShellMode::Client, None), &out);
        assert!(result.is_ok(), "{:?}", result.err().map(|f| f.message));
        assert!(!out.join("src/orbital_app/generated/program.json").exists());
        let _ = std::fs::remove_dir_all(out);
    }

    #[test]
    fn a_served_program_is_written_as_the_generated_program() {
        let out = scratch("served");
        PythonProgram
            .generate(&request(ShellMode::Full, Some(served("ServedApp"))), &out)
            .unwrap();
        let text =
            std::fs::read_to_string(out.join("src/orbital_app/generated/program.json")).unwrap();
        let program: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(program["name"], "ServedApp");
        assert!(out
            .join("src/orbital_app/generated/language.json")
            .is_file());
        let _ = std::fs::remove_dir_all(out);
    }
}
