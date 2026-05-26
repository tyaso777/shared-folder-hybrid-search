use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSchema {
    pub dataset_id: String,
    pub primary_key: String,
    pub text_fields: Vec<String>,
    #[serde(default)]
    pub display_fields: Vec<String>,
    #[serde(default)]
    pub full_text_fields: Vec<String>,
    #[serde(default)]
    pub source_uri_field: Option<String>,
    #[serde(default)]
    pub source_label_field: Option<String>,
    #[serde(default)]
    pub filter_fields: BTreeMap<String, FilterField>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterField {
    #[serde(rename = "type")]
    pub field_type: FilterType,
    pub label: Option<String>,
    #[serde(default = "default_filter_ui")]
    pub ui: FilterUi,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FilterType {
    Keyword,
    Number,
    Date,
    Boolean,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FilterUi {
    Select,
    MultiSelect,
    DateRange,
    NumberRange,
    Checkbox,
}

fn default_filter_ui() -> FilterUi {
    FilterUi::Select
}

#[derive(Debug, Clone)]
pub struct PreparedRecord {
    pub record_id: String,
    pub searchable_text: String,
    pub payload_json: String,
    pub display_json: String,
    pub filters: BTreeMap<String, Value>,
    pub row_hash: String,
}

#[derive(Debug, Error)]
pub enum SchemaError {
    #[error("schema.dataset_id is required")]
    MissingDatasetId,
    #[error("schema.primary_key is required")]
    MissingPrimaryKey,
    #[error("schema.text_fields must not be empty")]
    MissingTextFields,
    #[error("schema field `{0}` is referenced but not present in record")]
    MissingField(String),
    #[error("record must be a flat JSON object")]
    NotFlatObject,
    #[error("record `{0}` has empty searchable text")]
    EmptySearchableText(String),
    #[error("duplicate primary key `{0}`")]
    DuplicatePrimaryKey(String),
    #[error("field `{field}` has unsupported nested value")]
    NestedValue { field: String },
    #[error("filter field `{field}` has invalid value for type `{expected:?}`")]
    InvalidFilterValue { field: String, expected: FilterType },
}

impl DatasetSchema {
    pub fn validate(&self) -> Result<(), SchemaError> {
        if self.dataset_id.trim().is_empty() {
            return Err(SchemaError::MissingDatasetId);
        }
        if self.primary_key.trim().is_empty() {
            return Err(SchemaError::MissingPrimaryKey);
        }
        if self.text_fields.is_empty() {
            return Err(SchemaError::MissingTextFields);
        }
        Ok(())
    }

    pub fn prepare_records(&self, values: Vec<Value>) -> Result<Vec<PreparedRecord>, SchemaError> {
        self.validate()?;
        let mut seen = BTreeSet::new();
        let mut out = Vec::with_capacity(values.len());

        for value in values {
            let object = value.as_object().ok_or(SchemaError::NotFlatObject)?;
            for (key, value) in object {
                if matches!(value, Value::Array(_) | Value::Object(_)) {
                    return Err(SchemaError::NestedValue { field: key.clone() });
                }
            }

            let id_value = object
                .get(&self.primary_key)
                .ok_or_else(|| SchemaError::MissingField(self.primary_key.clone()))?;
            let record_id = scalar_to_string(id_value)
                .ok_or_else(|| SchemaError::MissingField(self.primary_key.clone()))?;
            if !seen.insert(record_id.clone()) {
                return Err(SchemaError::DuplicatePrimaryKey(record_id));
            }

            let mut text_parts = Vec::new();
            for field in &self.text_fields {
                let value = object
                    .get(field)
                    .ok_or_else(|| SchemaError::MissingField(field.clone()))?;
                if let Some(text) = scalar_to_string(value) {
                    if !text.trim().is_empty() {
                        text_parts.push(text);
                    }
                }
            }
            let searchable_text = text_parts.join("\n");
            if searchable_text.trim().is_empty() {
                return Err(SchemaError::EmptySearchableText(record_id));
            }

            for field in &self.full_text_fields {
                object
                    .get(field)
                    .ok_or_else(|| SchemaError::MissingField(field.clone()))?;
            }
            if let Some(field) = &self.source_uri_field {
                object
                    .get(field)
                    .ok_or_else(|| SchemaError::MissingField(field.clone()))?;
            }
            if let Some(field) = &self.source_label_field {
                object
                    .get(field)
                    .ok_or_else(|| SchemaError::MissingField(field.clone()))?;
            }

            let mut display = serde_json::Map::new();
            for field in &self.display_fields {
                let value = object
                    .get(field)
                    .ok_or_else(|| SchemaError::MissingField(field.clone()))?;
                display.insert(field.clone(), value.clone());
            }

            let mut filters = BTreeMap::new();
            for (field, spec) in &self.filter_fields {
                let value = object
                    .get(field)
                    .ok_or_else(|| SchemaError::MissingField(field.clone()))?;
                validate_filter_value(field, value, &spec.field_type)?;
                filters.insert(field.clone(), value.clone());
            }

            let payload_json = serde_json::to_string(&value).expect("serialize JSON value");
            let display_json =
                serde_json::to_string(&Value::Object(display)).expect("serialize display JSON");
            let row_hash = sha256_hex(payload_json.as_bytes());
            out.push(PreparedRecord {
                record_id,
                searchable_text,
                payload_json,
                display_json,
                filters,
                row_hash,
            });
        }

        Ok(out)
    }
}

fn scalar_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        Value::Null => Some(String::new()),
        Value::Array(_) | Value::Object(_) => None,
    }
}

fn validate_filter_value(
    field: &str,
    value: &Value,
    expected: &FilterType,
) -> Result<(), SchemaError> {
    let ok = match expected {
        FilterType::Keyword => {
            matches!(value, Value::String(_) | Value::Number(_) | Value::Bool(_))
        }
        FilterType::Number => value.as_f64().is_some(),
        FilterType::Date => value.as_str().is_some_and(|s| !s.trim().is_empty()),
        FilterType::Boolean => value.as_bool().is_some(),
    };
    if ok {
        Ok(())
    } else {
        Err(SchemaError::InvalidFilterValue {
            field: field.to_string(),
            expected: expected.clone(),
        })
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}
