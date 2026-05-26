use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::Path;

use hnsw_rs::prelude::*;

#[derive(Debug, Clone)]
pub struct VectorHit {
    pub record_id: String,
    pub distance: f32,
}

pub struct HnswVectorIndex {
    dim: usize,
    hnsw: Hnsw<'static, f32, DistCosine>,
    ids: Vec<String>,
    vectors: Vec<Vec<f32>>,
}

impl HnswVectorIndex {
    pub fn build(dim: usize, items: Vec<(String, Vec<f32>)>) -> anyhow::Result<Self> {
        let expected = items.len().max(100);
        let hnsw = Hnsw::<f32, DistCosine>::new(16, expected, 16, 200, DistCosine {});
        let mut ids = Vec::with_capacity(items.len());
        let mut vectors = Vec::with_capacity(items.len());
        for (label, (id, vector)) in items.into_iter().enumerate() {
            if vector.len() != dim {
                anyhow::bail!("vector dimension mismatch for `{id}`");
            }
            hnsw.insert((&vector[..], label));
            ids.push(id);
            vectors.push(vector);
        }
        Ok(Self {
            dim,
            hnsw,
            ids,
            vectors,
        })
    }

    pub fn load(dir: &Path, dim: usize) -> anyhow::Result<Self> {
        let map = fs::read_to_string(dir.join("map.tsv"))?;
        let ids = map
            .lines()
            .filter_map(|line| line.split_once('\t').map(|(_, id)| id.to_string()))
            .collect::<Vec<_>>();
        let mut vectors = Vec::with_capacity(ids.len());
        let mut reader = fs::File::open(dir.join("vectors.bin"))?;
        loop {
            let mut len_buf = [0u8; 4];
            if reader.read_exact(&mut len_buf).is_err() {
                break;
            }
            let len = u32::from_le_bytes(len_buf) as usize;
            let mut bytes = vec![0u8; len * 4];
            reader.read_exact(&mut bytes)?;
            vectors.push(bytemuck::cast_slice::<u8, f32>(&bytes).to_vec());
        }
        if ids.len() != vectors.len() {
            anyhow::bail!("HNSW snapshot id/vector count mismatch");
        }
        let hnsw = Hnsw::<f32, DistCosine>::new(16, vectors.len().max(100), 16, 200, DistCosine {});
        for (label, vector) in vectors.iter().enumerate() {
            if vector.len() == dim {
                hnsw.insert((&vector[..], label));
            }
        }
        Ok(Self {
            dim,
            hnsw,
            ids,
            vectors,
        })
    }

    pub fn save(&self, dir: &Path) -> anyhow::Result<()> {
        fs::create_dir_all(dir)?;
        let mut map = fs::File::create(dir.join("map.tsv.tmp"))?;
        for (idx, id) in self.ids.iter().enumerate() {
            writeln!(map, "{idx}\t{id}")?;
        }
        let mut vecs = fs::File::create(dir.join("vectors.bin.tmp"))?;
        for vector in &self.vectors {
            vecs.write_all(&(vector.len() as u32).to_le_bytes())?;
            vecs.write_all(bytemuck::cast_slice(vector))?;
        }
        fs::rename(dir.join("map.tsv.tmp"), dir.join("map.tsv"))?;
        fs::rename(dir.join("vectors.bin.tmp"), dir.join("vectors.bin"))?;
        Ok(())
    }

    pub fn search(&self, query: &[f32], limit: usize) -> Vec<VectorHit> {
        if query.len() != self.dim || limit == 0 {
            return Vec::new();
        }
        let hits = self.hnsw.search(query, limit, limit.max(32));
        let mut seen = HashMap::new();
        for hit in hits {
            if let Some(id) = self.ids.get(hit.d_id) {
                seen.entry(id.clone()).or_insert(hit.distance as f32);
            }
        }
        let mut out = seen
            .into_iter()
            .map(|(record_id, distance)| VectorHit {
                record_id,
                distance,
            })
            .collect::<Vec<_>>();
        out.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out
    }
}
