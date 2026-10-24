//! The Python shell's host: the native kernel (the `orbital_kernel` extension) serves the program, so the generated
//! project is the served program (the app composed with `std-http-server`) and the language it runs against.

use orbital_core_base::generate::{Backend, GeneratedFile, GeneratedOutput, ShellMode};
use orbital_core_base::ir::OirModule;

/// The JSON texts of the served program and the language it runs against.
pub struct KernelProject {
    pub program: String,
    pub language: String,
}


/// Writes `generated/program.orb` and `generated/language.json` for a project that serves a program.
#[derive(Default)]
pub struct PythonBackend {
    pub kernel: Option<KernelProject>,
}

impl PythonBackend {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Backend for PythonBackend {
    fn generate(&self, _module: &OirModule) -> GeneratedOutput {
        let files = match &self.kernel {
            Some(project) => vec![
                GeneratedFile::new("program.orb", project.program.clone()),
                GeneratedFile::new("language.json", project.language.clone()),
            ],
            None => Vec::new(),
        };
        GeneratedOutput {
            files,
            warnings: vec![],
        }
    }

    fn generate_with_mode(&self, module: &OirModule, mode: ShellMode) -> GeneratedOutput {
        match mode {
            ShellMode::Client => GeneratedOutput {
                files: vec![],
                warnings: vec![],
            },
            ShellMode::Full | ShellMode::Server => self.generate(module),
        }
    }

    fn name(&self) -> &'static str {
        "python"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn client_mode_writes_nothing_even_with_a_program() {
        let backend = PythonBackend {
            kernel: Some(KernelProject {
                program: "{}".to_string(),
                language: "{}".to_string(),
            }),
        };
        let module = OirModule::new("app", "1.0.0");
        assert!(backend
            .generate_with_mode(&module, ShellMode::Client)
            .files
            .is_empty());
        assert_eq!(
            backend
                .generate_with_mode(&module, ShellMode::Server)
                .files
                .len(),
            2
        );
    }
}
