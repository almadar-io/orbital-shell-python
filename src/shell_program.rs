//! `orb-shell-python`: the Python shell as a program (`orbital_core::shell_protocol`).

use orbital_core_base::project::write_project;
use orbital_core_base::shell_protocol::{
    description, written_files, GenerateRequest, GenerateResponse, ShellDescription, ShellFailure, ShellProgram,
};
use std::path::Path;

pub struct PythonProgram;

impl ShellProgram for PythonProgram {
    fn describe(&self) -> ShellDescription {
        description("python", env!("CARGO_PKG_VERSION"), false, Vec::new())
    }

    fn generate(&self, request: &GenerateRequest, output: &Path) -> Result<GenerateResponse, ShellFailure> {
        let generated = write_project(&request.module, output, &crate::PythonShell::new(), &crate::PythonBackend::new(), &request.write)?;
        Ok(GenerateResponse {
            files: written_files(&generated),
            warnings: generated.warnings,
            next_steps: vec![
                format!("cd {}", output.display()),
                "python -m venv .venv && source .venv/bin/activate".to_string(),
                "pip install -e .".to_string(),
                "python -m orbital_app.main".to_string(),
            ],
        })
    }

    fn emit(&self, request: &GenerateRequest) -> Result<orbital_core_base::generate::GeneratedOutput, ShellFailure> {
        use orbital_core_base::generate::Backend as _;
        Ok(crate::PythonBackend::new().generate(&request.module))
    }
}
