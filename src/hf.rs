use std::path::PathBuf;

use anyhow::{Context, Result};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};

#[derive(Debug, Clone)]
pub(crate) struct HfFile {
    pub(crate) repo_id: String,
    pub(crate) path: String,
    pub(crate) revision: Option<String>,
}

pub(crate) fn parse_hf_uri(uri: &str) -> Option<HfFile> {
    let uri = uri.strip_prefix("hf://")?;
    let (path_part, revision) = match uri.rsplit_once('@') {
        Some((path, rev)) => (path, Some(rev.to_string())),
        None => (uri, None),
    };
    let mut parts = path_part.split('/');
    let owner = parts.next()?;
    let repo = parts.next()?;
    let rest: Vec<&str> = parts.collect();
    if rest.is_empty() {
        return None;
    }
    Some(HfFile {
        repo_id: format!("{owner}/{repo}"),
        path: rest.join("/"),
        revision,
    })
}

pub(crate) fn download_hf_file(file: &HfFile) -> Result<PathBuf> {
    let api = ApiBuilder::from_env().build().unwrap();
    let repo = match &file.revision {
        Some(rev) => Repo::with_revision(file.repo_id.clone(), RepoType::Model, rev.clone()),
        None => Repo::new(file.repo_id.clone(), RepoType::Model),
    };
    let repo = api.repo(repo);
    let local = repo
        .get(&file.path)
        .with_context(|| format!("failed to download hf://{}/{}", file.repo_id, file.path))?;
    Ok(local)
}

pub(crate) fn ensure_file_from_hf(path: &std::path::Path, uri: &str) -> Result<()> {
    if path.exists() {
        return Ok(());
    }
    let hf = parse_hf_uri(uri)
        .ok_or_else(|| anyhow::anyhow!("unsupported uri {uri} (expected hf://...)"))?;
    let local = download_hf_file(&hf)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::copy(&local, path)
        .with_context(|| format!("failed to copy {:?} to {:?}", local, path))?;
    Ok(())
}
