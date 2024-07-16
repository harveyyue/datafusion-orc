use std::any::Any;
use std::fmt::{self, Debug};
use std::sync::Arc;

use crate::projection::ProjectionMask;
use crate::ArrowReaderBuilder;
use arrow::error::ArrowError;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::datasource::physical_plan::{
    FileMeta, FileOpenFuture, FileOpener, FileScanConfig, FileStream,
};
use datafusion::error::Result;
use datafusion::execution::context::TaskContext;
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning,
    SendableRecordBatchStream,
};
use datafusion_physical_expr::{PhysicalSortExpr};

use futures_util::StreamExt;
use object_store::ObjectStore;

use super::object_store_reader::ObjectStoreReader;

#[derive(Debug, Clone)]
pub struct OrcExec {
    config: FileScanConfig,
    metrics: ExecutionPlanMetricsSet,
    projected_schema: SchemaRef,
    projected_output_ordering: Vec<Vec<PhysicalSortExpr>>,
}

impl OrcExec {
    pub fn new(config: FileScanConfig) -> Self {
        let metrics = ExecutionPlanMetricsSet::new();
        let (projected_schema, _, projected_output_ordering) = config.project();
        Self {
            config,
            metrics,
            projected_schema,
            projected_output_ordering,
        }
    }
}

impl DisplayAs for OrcExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> std::fmt::Result {
        write!(f, "OrcExec: ")?;
        self.config.fmt_as(t, f)
    }
}

impl ExecutionPlan for OrcExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.projected_schema)
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(self.config.file_groups.len())
    }

    fn output_ordering(&self) -> Option<&[PhysicalSortExpr]> {
        self.projected_output_ordering
            .first()
            .map(|ordering| ordering.as_slice())
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    // fn properties(&self) -> &PlanProperties {
    //     &self.properties
    // }

    fn execute(
        &self,
        partition_index: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let projection: Vec<_> = self
            .config
            .projection
            .as_ref()
            .map(|p| {
                // FileScanConfig::file_column_projection_indices
                p.iter()
                    .filter(|col_idx| **col_idx < self.config.file_schema.fields().len())
                    .copied()
                    .collect()
            })
            .unwrap_or_else(|| (0..self.config.file_schema.fields().len()).collect());

        let object_store = context
            .runtime_env()
            .object_store(&self.config.object_store_url)?;

        let opener = OrcOpener {
            _partition_index: partition_index,
            projection,
            batch_size: context.session_config().batch_size(),
            _limit: self.config.limit,
            _table_schema: self.config.file_schema.clone(),
            _metrics: self.metrics.clone(),
            object_store,
        };

        let stream = FileStream::new(&self.config, partition_index, opener, &self.metrics)?;
        Ok(Box::pin(stream))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

// TODO: make use of the unused fields (e.g. implement metrics)
struct OrcOpener {
    _partition_index: usize,
    projection: Vec<usize>,
    batch_size: usize,
    _limit: Option<usize>,
    _table_schema: SchemaRef,
    _metrics: ExecutionPlanMetricsSet,
    object_store: Arc<dyn ObjectStore>,
}

impl FileOpener for OrcOpener {
    fn open(&self, file_meta: FileMeta) -> Result<FileOpenFuture> {
        let reader =
            ObjectStoreReader::new(self.object_store.clone(), file_meta.object_meta.clone());
        let batch_size = self.batch_size;
        // Offset by 1 since index 0 is the root
        let projection = self.projection.iter().map(|i| i + 1).collect::<Vec<_>>();
        Ok(Box::pin(async move {
            let builder = ArrowReaderBuilder::try_new_async(reader)
                .await
                .map_err(ArrowError::from)?;
            let projection_mask =
                ProjectionMask::roots(builder.file_metadata().root_data_type(), projection);
            let reader = builder
                .with_batch_size(batch_size)
                .with_projection(projection_mask)
                .build_async();

            Ok(reader.boxed())
        }))
    }
}
