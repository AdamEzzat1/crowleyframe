use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};

use polars::prelude::*;

use crate::selectors::Selector;

#[pyclass]
pub struct Frame {
    pub df: DataFrame,
}

#[pymethods]
impl Frame {
    #[new]
    fn new() -> Self {
        Self {
            df: DataFrame::default(),
        }
    }

    /// Construct from a Python dict-of-lists:
    /// {"col1": [1,2,3], "col2": ["a","b","c"]}
    #[staticmethod]
    pub fn from_dict(data: &PyDict) -> PyResult<Self> {
        let mut series: Vec<Series> = Vec::new();

        // Iterate over key/value pairs in the dict
        for (key, value) in data {
            let name: String = key.extract()?;
            let list: &PyList = value.downcast()?;

            let len = list.len();
            if len == 0 {
                // default to Utf8 empty series
                series.push(Series::new(&name, &Vec::<String>::new()));
                continue;
            }

            let first = list.get_item(0)?;

            // is_instance_of returns bool in this pyo3 version
            if first.is_instance_of::<pyo3::types::PyInt>() {
                let mut vals = Vec::<i64>::with_capacity(len);
                for item in list.iter() {
                    vals.push(item.extract()?);
                }
                series.push(Series::new(&name, vals));
            } else if first.is_instance_of::<pyo3::types::PyFloat>() {
                let mut vals = Vec::<f64>::with_capacity(len);
                for item in list.iter() {
                    vals.push(item.extract()?);
                }
                series.push(Series::new(&name, vals));
            } else if first.is_instance_of::<pyo3::types::PyBool>() {
                let mut vals = Vec::<bool>::with_capacity(len);
                for item in list.iter() {
                    vals.push(item.extract()?);
                }
                series.push(Series::new(&name, vals));
            } else {
                // fallback: treat as string
                let mut vals = Vec::<String>::with_capacity(len);
                for item in list.iter() {
                    vals.push(item.str()?.to_str()?.to_string());
                }
                series.push(Series::new(&name, vals));
            }
        }

        let df = DataFrame::new(series).map_err(to_py_err)?;
        Ok(Self { df })
    }

    /// Select columns using crowleyframe.col selectors
    pub fn select(&self, selectors: &PyAny) -> PyResult<Self> {
        // selectors is a Python list of ColSelector objects
        let selectors_vec: Vec<&PyAny> = selectors.extract()?;
        let mut cols: Vec<String> = Vec::new();

        for sel in selectors_vec {
            let selector = Selector::from_py(sel)?;
            let mut expanded = selector.apply(&self.df).map_err(to_py_err)?;
            cols.append(&mut expanded);
        }
        cols.dedup();

        let new_df = self.df.select(&cols).map_err(to_py_err)?;
        Ok(Self { df: new_df })
    }

    /// Clean column names to snake_case-ish
    pub fn clean_names(&self) -> PyResult<Self> {
        let mut df = self.df.clone();
        let new_names: Vec<String> = df
            .get_column_names()
            .iter()
            .map(|name| clean_name(name))
            .collect();
        df.set_column_names(&new_names).map_err(to_py_err)?;
        Ok(Self { df })
    }

    /// Convert back to dict-of-lists for Python
    pub fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        use pyo3::IntoPy;

        let dict = PyDict::new(py);

        for col in self.df.get_columns() {
            let name = col.name();
            let mut vals: Vec<PyObject> = Vec::with_capacity(col.len());

            // Iterate over AnyValue and convert to Python types
            for av in col.iter() {
                let obj: PyObject = match av {
                    AnyValue::Int64(v) => v.into_py(py),
                    AnyValue::Int32(v) => (v as i64).into_py(py),
                    AnyValue::UInt64(v) => (v as i64).into_py(py),
                    AnyValue::UInt32(v) => (v as i64).into_py(py),
                    AnyValue::Float64(v) => v.into_py(py),
                    AnyValue::Float32(v) => (v as f64).into_py(py),
                    AnyValue::Boolean(v) => v.into_py(py),

                    // Polars 0.42 string variants
                    AnyValue::String(s) => s.into_py(py),
                    AnyValue::StringOwned(s) => s.as_str().into_py(py),

                    AnyValue::Null => py.None(),
                    _ => {
                        // Fallback: string representation
                        format!("{}", av).into_py(py)
                    }
                };
                vals.push(obj);
            }

            let list = PyList::new(py, vals);
            dict.set_item(name, list)?;
        }

        Ok(dict.into())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.df))
    }
}

fn to_py_err(e: PolarsError) -> pyo3::PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

fn clean_name(name: &str) -> String {
    let mut out = String::new();
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push('_');
        }
    }

    let mut compressed = String::new();
    let mut prev_underscore = false;
    for ch in out.chars() {
        if ch == '_' {
            if !prev_underscore {
                compressed.push(ch);
            }
            prev_underscore = true;
        } else {
            prev_underscore = false;
            compressed.push(ch);
        }
    }

    compressed.trim_matches('_').to_string()
}
