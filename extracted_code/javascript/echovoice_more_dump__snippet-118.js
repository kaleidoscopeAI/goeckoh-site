    fn is_mutag_format(&self, df: &DataFrame) -> bool {
        let required_columns = ["x", "edge_index", "edge_attr", "y"];
        required_columns.iter().all(|&col| df.column(col).is_ok()) && df.column("x").unwrap().dtype().is_list()
    }

    async fn convert_existing_mutag(&self, df: &DataFrame, path: &Path) -> Result<Vec<ProcessedGraph>, ProcessingError> {
        let mut graphs = Vec::new();

        for i in 0..df.height() {
            let x = self.extract_list_list_f64(df, "x", i)?;
            let edge_index = self.extract_list_list_usize(df, "edge_index", i)?;
            let edge_attr = self.extract_list_list_f64(df, "edge_attr", i)?;
            let y = self.extract_i32(df, "y", i)?;

            let mutag_entry = MutagEntry {
                x,
                edge_index,
                edge_attr,
                y,
            };

            let metadata = GraphMetadata {
                source_path: path.to_string_lossy().to_string(),
                data_type: DataType::Unknown,
                original_format: "MUTAG_Parquet".to_string(),
                num_nodes: mutag_entry.x.len(),
                num_edges: mutag_entry.edge_index.len() / 2,
                confidence_score: 1.0,
                chemical_properties: None,
                processing_timestamp: chrono::Utc::now(),
            };

            let processing_stats = ProcessingStats {
                processing_time_ms: 0,
                memory_used_kb: 0,
                warnings: Vec::new(),
                success: true,
                processor_used: "TabularProcessor".to_string(),
            };

            graphs.push(ProcessedGraph {
                id: Uuid::new_v4().to_string(),
                mutag_entry,
                metadata,
                processing_stats,
            });
        }

        Ok(graphs)
    }

    fn extract_list_list_f64(&self, df: &DataFrame, col_name: &str, row: usize) -> Result<Vec<Vec<f64>>, ProcessingError> {
        let col = df.column(col_name).map_err(|e| ProcessingError::DataError(e.to_string()))?;
        let val = col.get(row).map_err(|e| ProcessingError::DataError(e.to_string()))?;
        if let AnyValue::List(series) = val {
            let mut outer = Vec::new();
            for inner_val in series.iter() {
                if let AnyValue::List(inner_series) = inner_val {
                    let inner = inner_series.f64().unwrap().into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
                    outer.push(inner);
                }
            }
            Ok(outer)
        } else {
            Err(ProcessingError::InvalidFormat(format!("Invalid type for {}", col_name)))
        }
    }

    fn extract_list_list_usize(&self, df: &DataFrame, col_name: &str, row: usize) -> Result<Vec<Vec<usize>>, ProcessingError> {
        let col = df.column(col_name).map_err(|e| ProcessingError::DataError(e.to_string()))?;
        let val = col.get(row).map_err(|e| ProcessingError::DataError(e.to_string()))?;
        if let AnyValue::List(series) = val {
            let mut outer = Vec::new();
            for inner_val in series.iter() {
                if let AnyValue::List(inner_series) = inner_val {
                    let inner = inner_series.i64().unwrap().into_iter().map(|opt| opt.unwrap_or(0) as usize).collect();
                    outer.push(inner);
                }
            }
            Ok(outer)
        } else {
            Err(ProcessingError::InvalidFormat(format!("Invalid type for {}", col_name)))
        }
    }

    fn extract_i32(&self, df: &DataFrame, col_name: &str, row: usize) -> Result<i32, ProcessingError> {
        let col = df.column(col_name).map_err(|e| ProcessingError::DataError(e.to_string()))?;
        let val = col.get(row).map_err(|e| ProcessingError::DataError(e.to_string()))?;
        if let AnyValue::Int32(v) = val {
            Ok(v)
        } else if let AnyValue::Int64(v) = val {
            Ok(v as i32)
        } else {
            Err(ProcessingError::InvalidFormat(format!("Invalid type for {}", col_name)))
        }
    }

    async fn tabular_to_graph(&self, df: &DataFrame, path: &Path, start_time: Instant) -> Result<Vec<ProcessedGraph>, ProcessingError> {
        let num_rows = df.height();
        if num_rows == 0 {
            return Ok(Vec::new());
        }

        let column_names = df.get_column_names();
        let numeric_cols = column_names.iter().filter(|&name| df.column(name).unwrap().dtype().is_numeric()).cloned().collect::<Vec<_>>();

        let feature_dim = 11;
        let mut node_features = Vec::with_capacity(num_rows);
        for i in 0..num_rows {
            let mut features = Vec::with_capacity(numeric_cols.len());
            for name in &numeric_cols {
                let col = df.column(name).unwrap();
                let val = col.get(i).unwrap();
                let num_val = match val {
                    AnyValue::Float64(f) => f,
                    AnyValue::Float32(f) => f as f64,
                    AnyValue::Int64(i) => i as f64,
                    AnyValue::Int32(i) => i as f64,
                    AnyValue::UInt64(u) => u as f64,
                    AnyValue::UInt32(u) => u as f64,
                    _ => 0.0,
                };
                features.push(num_val);
            }
            // Pad or slice to fixed dim
            if features.len() > feature_dim {
                features.truncate(feature_dim);
            } else {
                features.resize(feature_dim, 0.0);
            }
            node_features.push(features);
        }

        // Create chain edges
        let mut edges = Vec::new();
        let mut edge_features = Vec::new();
        for i in 0..num_rows - 1 {
            edges.push(vec![i, i + 1]);
            edges.push(vec![i + 1, i]);
            edge_features.push(vec![1.0, 0.0, 0.0, 0.0]);
            edge_features.push(vec![1.0, 0.0, 0.0, 0.0]);
        }

        // Determine y
        let y = if let Ok(y_col) = df.column("y") {
            let mean = y_col.mean().unwrap_or(0.0);
            if mean > 0.5 { 1 } else { 0 }
        } else {
            0
        };

        let mutag_entry = MutagEntry {
            x: node_features,
            edge_index: edges,
            edge_attr: edge_features,
            y,
        };

        let metadata = GraphMetadata {
            source_path: path.to_string_lossy().to_string(),
            data_type: DataType::Unknown,
            original_format: "Tabular".to_string(),
            num_nodes: num_rows,
            num_edges: mutag_entry.edge_index.len() / 2,
            confidence_score: 0.8,
            chemical_properties: None,
            processing_timestamp: chrono::Utc::now(),
        };

        let processing_stats = ProcessingStats {
            processing_time_ms: start_time.elapsed().as_millis() as u64,
            memory_used_kb: 0,
            warnings: Vec::new(),
            success: true,
            processor_used: "TabularProcessor".to_string(),
        };

        Ok(vec![ProcessedGraph {
            id: Uuid::new_v4().to_string(),
            mutag_entry,
            metadata,
            processing_stats,
        }])
    }
