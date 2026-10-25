//! The kernel as a native Python extension. `Server` is the `orbital-wasm` server API (`orbital_wasm::stored`,
//! `api`), the same surface `orbital-node` exports, so a host loads any of them behind the same ports. Marshalling
//! only: every decision is `orbital_core::embed`'s. The GIL is released while the kernel runs. `serve` runs a whole served
//! program on the native host.

use orbital_wasm::api;
use orbital_wasm::stored::{ServerHandle, Slot};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::Arc;

fn fail(message: String) -> PyErr {
    PyRuntimeError::new_err(message)
}

/// Run a served program (the app composed with `std-http-server`) on the native host: boot it with its store and
/// ticks, mount `INIT` on every trait, and hold it until SIGINT or SIGTERM. The GIL is released while it runs.
#[pyfunction]
fn serve(py: Python<'_>, program_json: &str, catalogs_json: &str) -> PyResult<()> {
    let catalogs: orbital_shell_rust::host::boot::Catalogs =
        serde_json::from_str(catalogs_json).map_err(|e| fail(format!("parse catalogs_json: {e}")))?;
    py.allow_threads(|| {
        let env = std::env::vars().collect();
        let runtime = orbital_shell_rust::host::boot::boot_embedded(program_json, &env, catalogs).map_err(|e| e.to_string())?;
        orbital_shell_rust::host::run::mount_init(&runtime, orbital_shell_rust::host::run::ProtocolSide::Server).map_err(|e| format!("INIT: {e}"))?;
        let stop = orbital_shell_rust::host::run::stop_on_signal().map_err(|e| e.to_string())?;
        orbital_shell_rust::host::run::hold(&runtime, &stop);
        Ok(())
    })
    .map_err(fail)
}

/// The server role: `start` a request, answer the I/O it asks for, `resume` the flight until it is done.
#[pyclass]
struct Server {
    held: Slot<ServerHandle>,
}

impl Server {
    fn handle(&self) -> PyResult<Arc<ServerHandle>> {
        self.held.get().map_err(fail)
    }
}

#[pymethods]
impl Server {
    #[new]
    fn new(resolved_schema_json: &str) -> PyResult<Self> {
        Ok(Self {
            held: Slot::new(ServerHandle::plain(resolved_schema_json).map_err(fail)?),
        })
    }

    fn free(&self) {
        self.held.take();
    }

    fn os_allows(&self, requirement_json: &str) -> PyResult<bool> {
        self.handle()?
            .api()
            .os_allows(requirement_json)
            .map_err(fail)
    }

    fn start(&self, py: Python<'_>, orbital: &str, request_json: &str) -> PyResult<String> {
        let handle = self.handle()?;
        py.allow_threads(|| handle.start(orbital, request_json))
            .map_err(fail)
    }

    fn resume(&self, py: Python<'_>, flight: u64, replies_json: &str) -> PyResult<String> {
        let handle = self.handle()?;
        py.allow_threads(|| handle.resume(flight, replies_json))
            .map_err(fail)
    }

    fn resolve_rows(&self, py: Python<'_>, ref_json: &str, viewer_json: &str) -> PyResult<String> {
        let handle = self.handle()?;
        py.allow_threads(|| handle.resolve_rows(ref_json, viewer_json))
            .map_err(fail)
    }

    fn resume_resolve(&self, py: Python<'_>, flight: u64, replies_json: &str) -> PyResult<String> {
        let handle = self.handle()?;
        py.allow_threads(|| handle.resume_resolve(flight, replies_json))
            .map_err(fail)
    }

    fn abandon(&self, flight: u64) -> PyResult<bool> {
        Ok(self.handle()?.abandon(flight))
    }

    fn in_flight(&self) -> PyResult<usize> {
        Ok(self.handle()?.api().in_flight())
    }

    #[pyo3(signature = (trait_name, entity_id=None))]
    fn get_state(&self, trait_name: &str, entity_id: Option<&str>) -> PyResult<String> {
        Ok(self.handle()?.api().get_state(trait_name, entity_id))
    }

    fn set_managed_program(&self, program_json: &str) -> PyResult<()> {
        self.handle()?.api().set_managed_program(program_json).map_err(fail)
    }

    fn set_default_user(&self, user_json: &str) -> PyResult<()> {
        self.handle()?
            .api()
            .set_default_user(user_json)
            .map_err(fail)
    }

    fn mount(&self, traits_json: &str) -> PyResult<()> {
        self.handle()?.api().mount(traits_json).map_err(fail)
    }

    fn remount(&self, traits_json: &str) -> PyResult<()> {
        self.handle()?.api().remount(traits_json).map_err(fail)
    }

    fn set_messages(&self, locales_json: &str, catalogs_json: &str) -> PyResult<()> {
        self.handle()?
            .api()
            .set_messages(locales_json, catalogs_json)
            .map_err(fail)
    }

    fn set_capabilities(
        &self,
        entity_type: &str,
        query_json: &str,
        write_json: &str,
    ) -> PyResult<()> {
        self.handle()?
            .api()
            .set_capabilities(entity_type, query_json, write_json)
            .map_err(fail)
    }

    fn set_tick_mount(&self, traits_json: &str) -> PyResult<()> {
        self.handle()?
            .api()
            .set_tick_mount(traits_json)
            .map_err(fail)
    }

    fn last_error(&self) -> PyResult<String> {
        Ok(self.handle()?.api().last_error())
    }
}

#[pyfunction]
fn install_language(bundle_json: &str) -> PyResult<()> {
    api::install_language(bundle_json).map_err(fail)
}

#[pyfunction]
fn os_glob_matches(glob: &str, path: &str) -> bool {
    api::os_glob_matches(glob, path)
}

#[pymodule]
fn orbital_kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Server>()?;
    m.add_function(wrap_pyfunction!(serve, m)?)?;
    m.add_function(wrap_pyfunction!(install_language, m)?)?;
    m.add_function(wrap_pyfunction!(os_glob_matches, m)?)?;
    Ok(())
}
