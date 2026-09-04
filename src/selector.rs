use rand::prelude::IndexedRandom;

pub trait ModelSelector: Send {
    fn next_model(&mut self) -> Option<String>;
    fn has_more(&self) -> bool;
}

pub struct SequentialSelector {
    models: Vec<String>,
    index: usize,
}

impl SequentialSelector {
    pub fn new(models: Vec<String>) -> Self {
        Self { models, index: 0 }
    }
}

impl ModelSelector for SequentialSelector {
    fn next_model(&mut self) -> Option<String> {
        if self.index < self.models.len() {
            let model = self.models[self.index].clone();
            self.index += 1;
            Some(model)
        } else {
            None
        }
    }

    fn has_more(&self) -> bool {
        self.index < self.models.len()
    }
}

pub struct RandomSelector {
    remaining: Vec<(String, f64)>,
}

impl RandomSelector {
    pub fn uniform(models: Vec<String>) -> Self {
        let remaining = models.into_iter().map(|m| (m, 1.0)).collect();
        Self { remaining }
    }

    pub fn weighted(models: Vec<(String, f64)>) -> Self {
        Self { remaining: models }
    }
}

impl ModelSelector for RandomSelector {
    fn next_model(&mut self) -> Option<String> {
        if self.remaining.is_empty() {
            return None;
        }

        let total_weight: f64 = self.remaining.iter().map(|(_, w)| w).sum();
        if total_weight <= 0.0 {
            return None;
        }

        let mut rng = rand::rng();
        let indices: Vec<usize> = (0..self.remaining.len()).collect();
        let idx = *indices
            .choose_weighted(&mut rng, |&i| self.remaining[i].1 / total_weight)
            .unwrap_or(&0);

        let (model, _) = self.remaining.remove(idx);
        Some(model)
    }

    fn has_more(&self) -> bool {
        !self.remaining.is_empty()
    }
}

#[derive(Debug, Clone)]
pub enum ModelInput {
    Sequential(Vec<String>),
    Random(Vec<String>),
    Weighted(Vec<(String, f64)>),
}

impl ModelInput {
    /// Rejects an empty leg, so `"x/a,,x/b"` fails the same way it already does
    /// through `validate` and `resolve_endpoints` instead of silently becoming a
    /// two-leg chain.
    pub fn validate(&self) -> Result<(), crate::Error> {
        let models: Vec<&String> = match self {
            ModelInput::Sequential(models) | ModelInput::Random(models) => models.iter().collect(),
            ModelInput::Weighted(models) => models.iter().map(|(model, _)| model).collect(),
        };
        if models.is_empty() {
            return Err(crate::Error::InvalidModelList {
                reason: "value must not be empty".into(),
            });
        }
        if models.iter().any(|model| model.trim().is_empty()) {
            return Err(crate::Error::InvalidModelList {
                reason: "list contains empty entry".into(),
            });
        }
        Ok(())
    }

    pub fn into_selector(self) -> Box<dyn ModelSelector> {
        match self {
            ModelInput::Sequential(models) => Box::new(SequentialSelector::new(models)),
            ModelInput::Random(models) => Box::new(RandomSelector::uniform(models)),
            ModelInput::Weighted(models) => Box::new(RandomSelector::weighted(models)),
        }
    }
}

impl From<&str> for ModelInput {
    fn from(s: &str) -> Self {
        // Empty entries are kept so `validate` can reject them; dropping one here
        // would silently turn a typo into a shorter chain.
        let models: Vec<String> = s.split(',').map(|m| m.trim().to_string()).collect();
        ModelInput::Sequential(models)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_selector() {
        let mut sel = SequentialSelector::new(vec!["a".into(), "b".into()]);
        assert!(sel.has_more());
        assert_eq!(sel.next_model(), Some("a".into()));
        assert!(sel.has_more());
        assert_eq!(sel.next_model(), Some("b".into()));
        assert!(!sel.has_more());
        assert_eq!(sel.next_model(), None);
    }

    #[test]
    fn test_random_selector_exhausts() {
        let mut sel = RandomSelector::uniform(vec!["a".into(), "b".into(), "c".into()]);
        let mut seen = Vec::new();
        while let Some(m) = sel.next_model() {
            seen.push(m);
        }
        assert_eq!(seen.len(), 3);
        assert!(seen.contains(&"a".to_string()));
        assert!(seen.contains(&"b".to_string()));
        assert!(seen.contains(&"c".to_string()));
    }

    #[test]
    fn test_model_input_from_str() {
        let input = ModelInput::from("a, b, c");
        match input {
            ModelInput::Sequential(models) => {
                assert_eq!(models, vec!["a", "b", "c"]);
            }
            _ => panic!("expected Sequential"),
        }
    }
}
