"""
Import Tests
=============

Basic tests to verify all modules can be imported correctly.
"""

import pytest


class TestCoreImports:
    """Test core module imports."""

    def test_import_src(self):
        """Test main package import."""
        import src
        assert hasattr(src, "__version__")

    def test_import_utils(self):
        """Test utils imports."""
        from src.utils import get_logger, load_config
        from src.utils.exceptions import DeepfakeDetectionError

    def test_import_preprocessing(self):
        """Test preprocessing imports."""
        from src.preprocessing import (
            VideoProcessor,
            FaceDetector,
            AudioProcessor,
        )

    def test_import_features(self):
        """Test features imports."""
        from src.features import (
            VisualEncoder,
            TemporalModel,
            AudioEncoder,
            CorrelationAnalyzer,
        )

    def test_import_retrieval(self):
        """Test retrieval imports."""
        from src.retrieval import VectorStore, FeatureAugmenter

    def test_import_classifier(self):
        """Test classifier imports."""
        from src.classifier import DeepfakeClassifier

    def test_import_orchestration(self):
        """Test orchestration imports."""
        from src.orchestration import Task, DAG, LocalExecutor

    def test_import_pipeline(self):
        """Test pipeline imports."""
        from src.pipeline import (
            DeepfakeDetector,
            DetectionResult,
            TrainingPipeline,
        )

    def test_import_realtime(self):
        """Test realtime imports."""
        from src.realtime import (
            StreamProcessor,
            StreamConfig,
            AlertManager,
            Alert,
            AlertLevel,
        )


class TestUtilityFunctions:
    """Test utility function behavior."""

    def test_logger_creation(self):
        """Test logger can be created."""
        from src.utils import get_logger
        logger = get_logger("test")
        assert logger is not None

    def test_config_loading(self):
        """Test config loading."""
        from src.utils.config import Config
        config = Config({"test": {"key": "value"}})
        assert config.get("test.key") == "value"

    def test_exception_hierarchy(self):
        """Test exception hierarchy."""
        from src.utils.exceptions import (
            DeepfakeDetectionError,
            PreprocessingError,
            FeatureExtractionError,
            ClassificationError,
        )

        assert issubclass(PreprocessingError, DeepfakeDetectionError)
        assert issubclass(FeatureExtractionError, DeepfakeDetectionError)
        assert issubclass(ClassificationError, DeepfakeDetectionError)


class TestDataClasses:
    """Test data classes."""

    def test_detection_result(self):
        """Test DetectionResult dataclass."""
        from src.pipeline.detector import DetectionResult

        result = DetectionResult(
            video_path="test.mp4",
            is_fake=True,
            probability=0.85,
            confidence=0.9,
        )

        assert result.video_path == "test.mp4"
        assert result.is_fake is True
        assert result.label == "FAKE"
        assert "probability" in result.to_dict()

    def test_alert_dataclass(self):
        """Test Alert dataclass."""
        from src.realtime.alerter import Alert, AlertLevel

        alert = Alert(
            level=AlertLevel.WARNING,
            title="Test Alert",
            message="This is a test",
        )

        assert alert.level == AlertLevel.WARNING
        assert alert.alert_id  # Should be auto-generated
        assert "title" in alert.to_dict()

    def test_task_status(self):
        """Test TaskStatus enum."""
        from src.orchestration.task import TaskStatus

        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.SUCCESS.value == "success"
        assert TaskStatus.FAILED.value == "failed"


class TestDAG:
    """Test DAG functionality."""

    def test_dag_creation(self):
        """Test DAG can be created."""
        from src.orchestration.dag import DAG

        dag = DAG("test_pipeline")
        assert dag.dag_id == "test_pipeline"
        assert len(dag) == 0

    def test_dag_task_addition(self):
        """Test adding tasks to DAG."""
        from src.orchestration.dag import DAG
        from src.orchestration.task import Task

        dag = DAG("test_pipeline")
        task = Task(name="test_task", callable=lambda ctx: None)
        dag.add_task(task)

        assert "test_task" in dag
        assert len(dag) == 1

    def test_dag_validation(self):
        """Test DAG validation."""
        from src.orchestration.dag import DAG
        from src.orchestration.task import Task

        dag = DAG("test_pipeline")
        task1 = Task(name="task1", callable=lambda ctx: None)
        task2 = Task(name="task2", callable=lambda ctx: None, depends_on=["task1"])

        dag.add_task(task1)
        dag.add_task(task2)

        errors = dag.validate()
        assert len(errors) == 0

    def test_dag_execution_order(self):
        """Test DAG execution order."""
        from src.orchestration.dag import DAG
        from src.orchestration.task import Task

        dag = DAG("test_pipeline")
        dag.add_task(Task(name="a", callable=lambda ctx: None))
        dag.add_task(Task(name="b", callable=lambda ctx: None, depends_on=["a"]))
        dag.add_task(Task(name="c", callable=lambda ctx: None, depends_on=["b"]))

        order = dag.get_execution_order()
        assert order.index("a") < order.index("b") < order.index("c")
