//! Module for parsing .nb files to create custom neuron models
//!
//! This module provides functionality to parse neuron model definitions from .nb files
//! and generate corresponding Rust code at compile time using procedural macros.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Represents a parsed neuron model from a .nb file
#[derive(Debug, Clone)]
pub struct NbNeuronModel {
    pub name: String,
    pub variables: HashMap<String, f32>,
    pub on_spike: Vec<String>,
    pub spike_detection: String,
    pub on_iteration: Vec<String>,
    pub ion_channels: HashMap<String, NbIonChannel>,
}

/// Represents a parsed ion channel from a .nb file
#[derive(Debug, Clone)]
pub struct NbIonChannel {
    pub name: String,
    pub variables: HashMap<String, f32>,
    pub gating_vars: Vec<String>,
    pub on_iteration: Vec<String>,
}

/// Parse a .nb file into a neuron model
pub fn parse_nb_file<P: AsRef<Path>>(path: P) -> Result<NbNeuronModel, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    parse_nb_content(&content)
}

/// Parse .nb file content into a neuron model
pub fn parse_nb_content(content: &str) -> Result<NbNeuronModel, Box<dyn std::error::Error>> {
    let mut lines = content.lines().map(|l| l.trim()).filter(|l| !l.is_empty() && !l.starts_with('#'));

    let mut neuron_model = NbNeuronModel {
        name: String::new(),
        variables: HashMap::new(),
        on_spike: Vec::new(),
        spike_detection: String::new(),
        on_iteration: Vec::new(),
        ion_channels: HashMap::new(),
    };

    let mut current_section = String::new();
    let mut current_ion_channel: Option<NbIonChannel> = None;

    for line in lines {
        if line.starts_with('[') && line.ends_with(']') {
            // New section
            current_section = line[1..line.len()-1].to_string();

            if current_section == "neuron" {
                // Parse neuron type
                if let Some(next_line) = lines.next() {
                    if next_line.starts_with("type:") {
                        neuron_model.name = next_line[5..].trim().to_string();
                    }
                }
            } else if current_section == "ion_channel" {
                current_ion_channel = Some(NbIonChannel {
                    name: String::new(),
                    variables: HashMap::new(),
                    gating_vars: Vec::new(),
                    on_iteration: Vec::new(),
                });
            }
        } else if !current_section.is_empty() {
            // Parse content based on section
            if current_section == "neuron" {
                if line.starts_with("vars:") {
                    // Parse variables
                    let vars_str = &line[5..];
                    for var in vars_str.split(',') {
                        let parts: Vec<&str> = var.split('=').map(|s| s.trim()).collect();
                        if parts.len() == 2 {
                            if let Ok(val) = parts[1].parse::<f32>() {
                                neuron_model.variables.insert(parts[0].to_string(), val);
                            }
                        }
                    }
                } else if line.starts_with("on_spike:") {
                    neuron_model.on_spike.push(line[8..].trim().to_string());
                } else if line.starts_with("spike_detection:") {
                    neuron_model.spike_detection = line[15..].trim().to_string();
                } else if line.starts_with("on_iteration:") {
                    neuron_model.on_iteration.push(line[13..].trim().to_string());
                }
            } else if current_section == "ion_channel" {
                if let Some(ref mut channel) = current_ion_channel {
                    if line.starts_with("type:") {
                        channel.name = line[5..].trim().to_string();
                    } else if line.starts_with("vars:") {
                        let vars_str = &line[5..];
                        for var in vars_str.split(',') {
                            let parts: Vec<&str> = var.split('=').map(|s| s.trim()).collect();
                            if parts.len() == 2 {
                                if let Ok(val) = parts[1].parse::<f32>() {
                                    channel.variables.insert(parts[0].to_string(), val);
                                }
                            }
                        }
                    } else if line.starts_with("gating_vars:") {
                        let vars_str = &line[6..];
                        for var in vars_str.split(',') {
                            channel.gating_vars.push(var.trim().to_string());
                        }
                    } else if line.starts_with("on_iteration:") {
                        channel.on_iteration.push(line[13..].trim().to_string());
                    }
                }
            } else if current_section == "end" {
                // End of section
                if let Some(channel) = current_ion_channel.take() {
                    neuron_model.ion_channels.insert(channel.name.clone(), channel);
                }
                current_section = String::new();
            }
        }
    }

    Ok(neuron_model)
}

/// Generate Rust code for a neuron model from the parsed .nb model
pub fn generate_neuron_code(model: &NbNeuronModel) -> String {
    let mut code = String::new();

    code.push_str(&format!("
#[derive(Debug, Clone)]
pub struct {}Neuron {{
", model.name));

    // Add fields based on variables
    for (var, default) in &model.variables {
        code.push_str(&format!("    pub {}: f32, // {}\n", var, default));
    }

    code.push_str("    pub dt: f32,\n");
    code.push_str("    pub is_spiking: bool,\n");
    code.push_str("    pub last_firing_time: Option<usize>,\n");

    // Add ion channels
    for (name, channel) in &model.ion_channels {
        code.push_str(&format!("    pub {}: {}Channel,\n", name.to_lowercase(), name));
    }

    code.push_str("}\n\n");

    // Add Default impl
    code.push_str(&format!("
impl Default for {}Neuron {{
    fn default() -> Self {{
        {}Neuron {{
", model.name, model.name));

    for (var, default) in &model.variables {
        code.push_str(&format!("            {}: {},\n", var, default));
    }

    code.push_str("            dt: 0.1,\n");
    code.push_str("            is_spiking: false,\n");
    code.push_str("            last_firing_time: None,\n");

    for (name, _) in &model.ion_channels {
        code.push_str(&format!("            {}: {}Channel::default(),\n", name.to_lowercase(), name));
    }

    code.push_str("        }\n    }\n}\n\n");

    // Add IterateAndSpike impl
    code.push_str(&format!("
impl IterateAndSpike for {}Neuron {{
    fn iterate_and_spike(&mut self, input_current: f32) -> bool {{
        // Generated from on_iteration
", model.name));

    for iter_code in &model.on_iteration {
        code.push_str(&format!("        {}\n", iter_code));
    }

    code.push_str("
        // Generated from spike_detection
        self.is_spiking = ");
    code.push_str(&model.spike_detection);
    code.push_str(";\n");

    code.push_str("
        if self.is_spiking {\n");
    for spike_code in &model.on_spike {
        code.push_str(&format!("            {}\n", spike_code));
    }
    code.push_str("        }\n");

    code.push_str("        self.is_spiking\n    }\n}\n");

    code
}

/// Macro to load and generate neuron model from .nb file at compile time
#[macro_export]
macro_rules! nb_neuron {
    ($file:expr) => {
        {
            use spiking_neural_networks::nb_parsing::parse_nb_file;
            let model = parse_nb_file($file).expect("Failed to parse .nb file");
            generate_neuron_code(&model)
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_nb_file() {
        let content = r#"
[neuron]
    type: BasicIntegrateAndFire
    vars: e = 0, v_reset = -75, v_th = -55
    on_spike:
        v = v_reset
    spike_detection: v >= v_th
    on_iteration:
        dv/dt = (v - e) + i
[end]
"#;

        let model = parse_nb_content(content).unwrap();
        assert_eq!(model.name, "BasicIntegrateAndFire");
        assert_eq!(model.variables.get("e"), Some(&0.0));
        assert_eq!(model.variables.get("v_reset"), Some(&-75.0));
        assert_eq!(model.variables.get("v_th"), Some(&-55.0));
        assert_eq!(model.on_spike, vec!["v = v_reset"]);
        assert_eq!(model.spike_detection, "v >= v_th");
        assert_eq!(model.on_iteration, vec!["dv/dt = (v - e) + i"]);
    }
}