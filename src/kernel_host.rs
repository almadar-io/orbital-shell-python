//! The Python shell's host: the native kernel (the `orbital_kernel` extension) serves the program, so the generated
//! project is the served program (the app composed with `std-http-server`) and the language it runs against.

use orbital_core_base::generate::{Backend, GeneratedFile, GeneratedOutput, ShellMode};
use orbital_core_base::ir::OirModule;

/// The JSON texts of the served program and the language it runs against.
pub struct KernelProject {
    pub program: String,
    pub language: String,
}


/// Writes the generated models (`entities.py`, `events.py`) and, for a project that serves a program, `program.orb` and `language.json`.
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
    fn generate(&self, module: &OirModule) -> GeneratedOutput {
        let mut files = vec![
            crate::models::package_file(),
            crate::models::entities_file(module),
            crate::models::events_file(module),
        ];
        if let Some(project) = &self.kernel {
            files.push(GeneratedFile::new("program.orb", project.program.clone()));
            files.push(GeneratedFile::new("language.json", project.language.clone()));
        }
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
            5
        );
    }
}
