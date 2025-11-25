use pyo3::prelude::*;
use polars::prelude::*;
use regex::Regex;

/// A single selector coming from Python (`col.*`)
pub enum Selector {
    Name(String),
    StartsWith(String),
    EndsWith(String),
    Contains(String),
    Matches(String),
    WhereNumeric,
    WhereString,
}

impl Selector {
    /// Parse Python selector object → Rust enum
    pub fn from_py(obj: &PyAny) -> PyResult<Self> {
        let kind: String = obj.getattr("kind")?.extract()?;

        match kind.as_str() {
            "name" => {
                let v: String = obj.getattr("value")?.extract()?;
                Ok(Selector::Name(v))
            }
            "starts_with" => {
                let v: String = obj.getattr("value")?.extract()?;
                Ok(Selector::StartsWith(v))
            }
            "ends_with" => {
                let v: String = obj.getattr("value")?.extract()?;
                Ok(Selector::EndsWith(v))
            }
            "contains" => {
                let v: String = obj.getattr("value")?.extract()?;
                Ok(Selector::Contains(v))
            }
            "matches" => {
                let v: String = obj.getattr("value")?.extract()?;
                Ok(Selector::Matches(v))
            }
            "where_numeric" => Ok(Selector::WhereNumeric),
            "where_string" => Ok(Selector::WhereString),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown selector kind: {}",
                kind
            ))),
        }
    }

    /// Apply selector to a Polars DataFrame → matching column names
    pub fn apply(&self, df: &DataFrame) -> PolarsResult<Vec<String>> {
        let cols = df.get_column_names();
        let mut out = Vec::new();

        match self {
            Selector::Name(name) => {
                if cols.contains(&name.as_str()) {
                    out.push(name.clone());
                }
            }

            Selector::StartsWith(prefix) => {
                for c in cols {
                    if c.starts_with(prefix) {
                        out.push(c.to_string());
                    }
                }
            }

            Selector::EndsWith(suffix) => {
                for c in cols {
                    if c.ends_with(suffix) {
                        out.push(c.to_string());
                    }
                }
            }

            Selector::Contains(sub) => {
                for c in cols {
                    if c.contains(sub) {
                        out.push(c.to_string());
                    }
                }
            }

            Selector::Matches(pat) => {
                let re = Regex::new(pat).map_err(|e| {
                    PolarsError::ComputeError(format!("Invalid regex: {}", e).into())
                })?;

                for c in cols {
                    if re.is_match(c) {
                        out.push(c.to_string());
                    }
                }
            }

            Selector::WhereNumeric => {
                for field in df.schema().iter_fields() {
                    match field.data_type() {
                        DataType::Int8
                        | DataType::Int16
                        | DataType::Int32
                        | DataType::Int64
                        | DataType::UInt8
                        | DataType::UInt16
                        | DataType::UInt32
                        | DataType::UInt64
                        | DataType::Float32
                        | DataType::Float64 => {
                            out.push(field.name().to_string());
                        }
                        _ => {}
                    }
                }
            }

            Selector::WhereString => {
                for field in df.schema().iter_fields() {
                    match field.data_type() {
                        DataType::String | DataType::Categorical(..) => {
                            out.push(field.name().to_string());
                        }
                        _ => {}
                    }
                }
            }
        }

        Ok(out)
    }
}
