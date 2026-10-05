"""
M1-Unemployment Debt Recovery SOP — Pydantic v2 Models
Generated from OWL ontology: m1-unemployment-sop

本版为「纯本体层」精简版：移除所有实例层对象（Call / Debtor / Agent /
Account / FinancialHealthProfile / CallOutcome 三子类），只保留 SOP 骨架，
并把通话级信号（RFD、结果、合规）上移到本体层节点上。
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# 1. Edge Helper — marks fields as graph relationships
# =============================================================================


def Edge(label: str, default: Any = None, required: bool = False, **kwargs):
    """
    Mark a field as a graph edge. Optional by default — pass required=True
    for edges that must be present in every valid extraction.
    """
    if required:
        return Field(..., json_schema_extra={"edge_label": label}, **kwargs)
    return Field(default, json_schema_extra={"edge_label": label}, **kwargs)


# =============================================================================
# 2. Enums — constrained vocabularies from OWL individuals
# =============================================================================


class MasterSOPStageEnum(str, Enum):
    """The 7 core stages in the master collection pipeline."""

    STAGE_1_OPENING = "Stage_1_Opening"
    STAGE_2_RPC = "Stage_2_RPC"
    STAGE_3_INQUIRY = "Stage_3_Inquiry"
    STAGE_4_APPLY_PRESSURE = "Stage_4_ApplyPressure"
    STAGE_5_WORKOUT_SOLUTION = "Stage_5_WorkoutSolution"
    STAGE_6_PUSH_PTP = "Stage_6_PushPTP"
    STAGE_7_CLOSING = "Stage_7_Closing"


class SubStageEnum(str, Enum):
    """M1-Unemployment sub-flow stages under Stage 3."""

    SUBSTAGE_3_1_EMPATHY_PIVOT = "SubStage_3_1_EmpathyPivot"
    SUBSTAGE_3_2_FINANCIAL_PROFILING = "SubStage_3_2_FinancialProfiling"
    SUBSTAGE_3_3_CARD_UTILITY = "SubStage_3_3_CardUtility"


class RFDEnum(str, Enum):
    """Reason for Delinquency — root cause categories."""

    INVOLUNTARY_UNEMPLOYMENT = "RFD_InvoluntaryUnemployment"


class ComplianceArtifactEnum(str, Enum):
    """Compliance checkpoints that must be logged during a call."""

    REQ_VERIFICATION = "Req_Verification"
    REQ_FIRST_CONTACT_DISCLOSURE = "Req_Disclosure"
    REQ_EXPLICIT_CONSENT = "Req_ExplicitConsent"


class ObjectionCategoryEnum(str, Enum):
    """Categories of objections raised by the debtor during a call."""

    FINANCIAL_HARDSHIP = "Objection_Financial_Hardship"
    DISPUTE_FEE_CONCERN = "Objection_Dispute_Fee_Concern"
    THIRD_PARTY_APPROVAL = "Objection_Third_Party_Approval"
    CALL_FREQUENCY_COMPLAINT = "Objection_Call_Frequency_Complaint"
    OUTRIGHT_REFUSAL = "Objection_Outright_Refusal"


class CallOutcomeEnum(str, Enum):
    """
    Outcomes a stage can yield.

    Attached to MasterSOPStage.yields_outcome / SubStage.yields_outcome
    so outcome signals live on the ontology layer instead of a Call object.
    """

    PROMISE_TO_PAY = "PromiseToPay"
    FORBEARANCE_AGREEMENT = "ForbearanceAgreement"
    MICRO_COMMITMENT = "MicroCommitment"


# =============================================================================
# 3. Canonical node models — BECOME GRAPH NODES
# =============================================================================


class ComplianceArtifact(BaseModel):
    """
    Mandatory regulatory or audit proof logged during a call.

    Canonical node.  Node ID: ComplianceArtifact__{chunk_id}__{artifact_id.value}
    """

    model_config = ConfigDict(graph_id_fields=["artifact_id"])

    artifact_id: ComplianceArtifactEnum = Field(
        ..., description="Which compliance checkpoint was satisfied."
    )
    label: Optional[str] = Field(
        None, description="Human-readable label for the compliance artifact."
    )


class MasterSOPStage(BaseModel):
    """
    One of the 7 core stages in the master collection pipeline.

    Canonical node.  Node ID: MasterSOPStage__{chunk_id}__{stage_id.value}

    Edges emitted from this node:
        next_master_stage        → MasterSOPStage
        branches_to_sub_stage    → SubStage
        requires_compliance[i]   → ComplianceArtifact
        handles_rfd[i]           → RFD (property or canonical node)
        yields_outcome[i]        → CallOutcome (property or canonical node)
    """

    model_config = ConfigDict(graph_id_fields=["stage_id"])

    stage_id: MasterSOPStageEnum = Field(
        ..., description="Identifier of this master SOP stage."
    )
    label: Optional[str] = Field(None, description="Descriptive label of the stage.")
    next_master_stage: Optional[MasterSOPStageEnum] = Edge(
        label="nextMasterStage",
        description="The subsequent master stage in the linear pipeline.",
    )
    branches_to_sub_stage: Optional[SubStageEnum] = Edge(
        label="branchesToSubStage",
        description="Sub-flow this master stage branches into (only Stage 3 typically).",
    )
    requires_compliance: list[ComplianceArtifactEnum] = Field(
        default_factory=list,
        description="Compliance artifacts required at this stage.",
        json_schema_extra={"edge_label": "requiresCompliance"},
    )
    handles_rfd: list[RFDEnum] = Field(
        default_factory=list,
        description=(
            "RFD categories this stage typically handles. "
            "For M1-unemployment, use ['RFD_InvoluntaryUnemployment']."
        ),
        json_schema_extra={"edge_label": "handlesRFD"},
    )
    yields_outcome: list[CallOutcomeEnum] = Field(
        default_factory=list,
        description=(
            "Outcomes this stage typically yields: "
            "'PromiseToPay', 'ForbearanceAgreement', 'MicroCommitment'."
        ),
        json_schema_extra={"edge_label": "yieldsOutcome"},
    )


class SubStage(BaseModel):
    """
    Specialized dialogue sub-states within the M1-Unemployment branch.

    Canonical node.  Node ID: SubStage__{chunk_id}__{substage_id.value}

    Edges emitted from this node:
        next_sub_stage           → SubStage
        rejoins_master_stage     → MasterSOPStage
        handles_rfd[i]           → RFD
        yields_outcome[i]        → CallOutcome
    """

    model_config = ConfigDict(graph_id_fields=["substage_id"])

    substage_id: SubStageEnum = Field(..., description="Identifier of this sub-stage.")
    label: Optional[str] = Field(None, description="Descriptive label.")
    next_sub_stage: Optional[SubStageEnum] = Edge(
        label="nextSubStage",
        description="The subsequent sub-stage in the branch flow.",
    )
    rejoins_master_stage: Optional[MasterSOPStageEnum] = Edge(
        label="rejoinsMasterStage",
        description="Which master stage this sub-flow routes back into.",
    )
    handles_rfd: list[RFDEnum] = Field(
        default_factory=list,
        description="RFD categories this sub-stage typically handles.",
        json_schema_extra={"edge_label": "handlesRFD"},
    )
    yields_outcome: list[CallOutcomeEnum] = Field(
        default_factory=list,
        description="Outcomes this sub-stage typically yields.",
        json_schema_extra={"edge_label": "yieldsOutcome"},
    )


# =============================================================================
# 4. ObjectionCategory — kept for backward compatibility
# =============================================================================
# Prompt 目前不要求输出 objections；保留该类以便 prompt 后续扩展时无需改模型。
# 若要启用，在 prompt 中加入 objections 字段并重新构建 checkpoint。


class ObjectionCategory(BaseModel):
    """
    A category of objection raised during the call.

    Node ID: ObjectionCategory__{chunk_id}__{objection_id.value}

    Edge emitted from this node:
        triggered_in_stage → MasterSOPStage
    """

    model_config = ConfigDict(graph_id_fields=["objection_id"])

    objection_id: ObjectionCategoryEnum = Field(
        ..., description="Which objection category was raised."
    )
    label: Optional[str] = Field(
        None, description="Descriptive label for the objection."
    )
    triggered_in_stage: Optional[MasterSOPStageEnum] = Edge(
        label="triggeredInStage",
        description="Master SOP stage where this objection was triggered.",
    )


# =============================================================================
# 5. Root Extraction Model — ontology layer only
# =============================================================================


class DebtCollectionExtraction(BaseModel):
    """
    Root model for LLM extraction from a single M1-Unemployment debt collection call.

    本版根模型**只有三个字段**，对应本体层的三类规范节点：
      * master_stages        — 主 SOP 阶段链
      * sub_stages           — M1-失业子流程
      * compliance_artifacts — 合规检查点

    通话级信号（RFD、结果、合规）已上移到各节点的 handles_rfd /
    yields_outcome / requires_compliance 字段上，因此不再需要 call 对象。
    """

    master_stages: list[MasterSOPStage] = Field(
        default_factory=list,
        description="Master SOP stages referenced or traversed during the call.",
    )
    sub_stages: list[SubStage] = Field(
        default_factory=list,
        description="Sub-stages traversed during the M1-Unemployment branch.",
    )
    compliance_artifacts: list[ComplianceArtifact] = Field(
        default_factory=list,
        description="Compliance artifacts referenced in the call.",
    )