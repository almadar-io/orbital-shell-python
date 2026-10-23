//! Python Shell Implementation
//!
//! Implements the Shell trait for Python projects that run a served program on the native kernel.

use orbital_core_base::shell::{Shell, ShellError};
use std::path::{Path, PathBuf};

/// Python shell: the Python host of a program served by the native kernel
pub struct PythonShell {
    template_path: PathBuf,
}

impl PythonShell {
    /// Create a new Python shell
    pub fn new() -> Self {
        let template_path = Self::find_template_path();
        Self { template_path }
    }

    /// Find template path at runtime using multiple strategies.
    ///
    /// Mirrors the native-shell lookup (android/ios/rust): env var,
    /// exe-relative, repo-root (this crate's `template/`),
    /// compile-time fallback. `python-shell` is a self-contained committed
    /// scaffold under `templates/`, like every other shell.
    fn find_template_path() -> PathBuf {
        // The one template lookup (`ORBITAL_SHELLS_DIR` → beside the binary → the checkout).
        orbital_core_base::renderer_packs::shell_template(
            "python-shell",
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")),
        )
    }

    /// Create with explicit template path
    pub fn with_template_path(template_path: impl Into<PathBuf>) -> Self {
        Self {
            template_path: template_path.into(),
        }
    }
}

impl Default for PythonShell {
    fn default() -> Self {
        Self::new()
    }
}

impl Shell for PythonShell {
    fn name(&self) -> &'static str {
        "python"
    }

    fn description(&self) -> &'static str {
        "Python host running the served program on the native kernel"
    }

    fn extensions(&self) -> &[&'static str] {
        &["py"]
    }

    fn template_path(&self) -> &Path {
        &self.template_path
    }

    fn generated_dir(&self) -> &str {
        "src/orbital_app/generated"
    }

    fn build_command(&self) -> &str {
        "pip install -e ."
    }

    fn dev_command(&self) -> &str {
        "python -m orbital_app"
    }

    fn test_command(&self) -> &str {
        "pytest"
    }

    fn init_project(&self, target_dir: &Path, project_name: &str) -> Result<(), ShellError> {
        // Copy template to target directory
        self.copy_template(target_dir)?;

        // Update pyproject.toml with project name
        let pyproject_path = target_dir.join("pyproject.toml");
        if pyproject_path.exists() {
            let content = std::fs::read_to_string(&pyproject_path)?;

            let updated = content.replace(
                "name = \"orbital-app\"",
                &format!("name = \"{}\"", project_name),
            );

            std::fs::write(&pyproject_path, updated)?;
        }

        Ok(())
    }

    fn copy_template(&self, output_dir: &Path) -> Result<(), ShellError> {
        if !self.template_path.exists() {
            return Err(ShellError::NotFound(
                self.template_path.display().to_string(),
            ));
        }

        copy_dir_recursive(&self.template_path, output_dir)?;
        Ok(())
    }

    fn validate_structure(&self, output_dir: &Path) -> Result<(), ShellError> {
        let required_files = ["pyproject.toml"];

        for file in &required_files {
            let path = output_dir.join(file);
            if !path.exists() {
                return Err(ShellError::InvalidStructure {
                    path: path.clone(),
                    message: format!("Missing required file: {}", file),
                });
            }
        }

        Ok(())
    }
}

/// Recursively copy a directory
fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<(), ShellError> {
    if !dst.exists() {
        std::fs::create_dir_all(dst)?;
    }

    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());

        // Skip generated directory, venv, and cache
        let file_name = entry.file_name();
        let file_name_str = file_name.to_string_lossy();
        if file_name_str == "generated"
            || file_name_str == "venv"
            || file_name_str == ".venv"
            || file_name_str == "__pycache__"
            || file_name_str == ".git"
            || file_name_str == ".pytest_cache"
            || file_name_str.ends_with(".egg-info")
        {
            continue;
        }

        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else {
            std::fs::copy(&src_path, &dst_path)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shell_properties() {
        let shell = PythonShell::new();
        assert_eq!(shell.name(), "python");
        assert_eq!(shell.extensions(), &["py"]);
        assert_eq!(shell.generated_dir(), "src/orbital_app/generated");
    }

    #[test]
    fn copied_scaffold_is_the_entry_point_without_the_retired_runtime() {
        let output = std::env::temp_dir().join(format!(
            "orbital_python_runtime_copy_{}",
            std::process::id()
        ));
        PythonShell::new().copy_template(&output).unwrap();
        let main = std::fs::read_to_string(output.join("src/orbital_app/main.py")).unwrap();
        assert!(main.contains("orbital_kernel"));
        assert!(!output.join("src/orbital_app/core").exists());
        assert!(!output.join("src/orbital_app/nn").exists());
        std::fs::remove_dir_all(output).unwrap();
    }
}
