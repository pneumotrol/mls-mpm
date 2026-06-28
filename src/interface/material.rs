//! Material-related data structures for the simulation interface.

use crate::Result;
use bon::Builder;
use color_eyre::eyre::eyre;
use std::{collections::HashMap, fmt::Display, ops::Deref};

/// A unique identifier for a material by name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MaterialName(String);

impl Display for MaterialName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for MaterialName {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

/// Supported material types and their physical parameters.
#[derive(Debug, Clone)]
pub enum MaterialKind {
    /// A fluid material defined by density and rheological properties.
    Fluid {
        /// Rest density of the fluid.
        density: f32,
        /// Specific heat ratio (gamma) for the equation of state.
        specific_heat_ratio: f32,
        /// Stiffness coefficient for the equation of state.
        stiffness: f32,
        /// Dynamic viscosity.
        viscosity: f32,
    },
    /// A purely elastic (Neo-Hookean solid) material.
    Elastic {
        /// Density of the material.
        density: f32,
        /// Shear modulus (mu) for elastic shear deformation.
        shear_modulus: f32,
        /// Bulk modulus (lambda) for volume preservation.
        bulk_modulus: f32,
    },
}

/// Configuration for a single material definition.
#[derive(Debug, Clone, Builder)]
pub struct MaterialDescriptor {
    pub(crate) name: MaterialName,
    pub(crate) kind: MaterialKind,
}

/// A unique numeric ID assigned to each material for GPU lookup.
#[derive(Debug, Clone, Copy)]
pub struct MaterialId(u32);

impl Deref for MaterialId {
    type Target = u32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// A registry for materials defined in the simulation.
///
/// Maps material names to their numeric IDs and physical properties.
#[derive(Debug, Clone, Default)]
pub struct MaterialDict(HashMap<MaterialName, (MaterialId, MaterialKind)>);

impl Deref for MaterialDict {
    type Target = HashMap<MaterialName, (MaterialId, MaterialKind)>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl MaterialDict {
    /// Creates an empty material dictionary.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a new material in the dictionary.
    ///
    /// # Errors
    ///
    /// Returns an error if a material with the same name already exists.
    pub fn register(&mut self, material: MaterialDescriptor) -> Result<&mut Self> {
        if !self.0.contains_key(&material.name) {
            self.0.insert(
                material.name,
                (MaterialId(self.0.len().try_into()?), material.kind),
            );

            Ok(self)
        } else {
            Err(eyre!(r#"Material name `{}` conflicts"#, material.name))
        }
    }

    /// Returns the `MaterialName` for a given string if it exists in the dictionary.
    pub fn get(&self, material_name: &str) -> Result<MaterialName> {
        let name = MaterialName(material_name.to_string());

        if self.0.contains_key(&name) {
            Ok(name)
        } else {
            Err(eyre!(r#"Material `{}` not found"#, material_name))
        }
    }
}
