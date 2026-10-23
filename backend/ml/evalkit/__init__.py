"""Leakage-safe ML evaluation toolkit (Phase 2). See docs/phase2/ML_EXPERIMENT_PROTOCOL.md."""
from .metrics import (aggregate_by_group, auc_stat, cluster_bootstrap, evaluate_binary,  # noqa: F401
                      expected_calibration_error, group_label_permutation_test, paired_cluster_bootstrap_diff,
                      ppv_npv_at_prevalence, regression_metrics, reliability_table, score_metrics,
                      threshold_metrics, wilson_interval)
from .nested import inner_select, nested_group_cv, naive_selection_oof, youden_threshold  # noqa: F401
from .splits import (LeakageError, assert_group_disjoint, assert_group_labels_consistent,  # noqa: F401
                     assert_no_duplicate_rows, assert_temporal_order, group_kfold, guarded,
                     leave_one_group_out, temporal_group_holdout)
