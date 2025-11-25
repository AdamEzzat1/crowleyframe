use pyo3::prelude::*;

mod frame;
mod selectors;

use frame::Frame;

#[pymodule]
fn _crowley(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Frame>()?;
    Ok(())
}

