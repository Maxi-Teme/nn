#[derive(Debug)]
pub enum NNError {
    FileSystem(std::io::Error),
    Serde(serde_json::Error),
}
