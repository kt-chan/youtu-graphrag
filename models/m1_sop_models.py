"""
M1-Unemployment Debt Recovery SOP — Pydantic v2 Models
Generated from OWL ontology: m1-unemployment-sop

Usage with LLM extraction (e.g., instructor / OpenAI structured outputs):
    from instructor import from_openai
    client = from_openai(OpenAI())
    result = client.chat.completions.create(
        response_model=DebtCollectionExtraction,
        messages=[...]
    )
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    REQ_MINI_MIRANDA = "Req_MiniMiranda"
    REQ_EXPLICIT_CONSENT = "Req_ExplicitConsent"


class CallOutcomeType(str, Enum):
    """Discriminator for CallOutcome subclasses."""

    PROMISE_TO_PAY = "PromiseToPay"
    FORBEARANCE_AGREEMENT = "ForbearanceAgreement"
    MICRO_COMMITMENT = "MicroCommitment"


# =============================================================================
# 3. Value Objects (Components) — deduplicated by content
# =============================================================================


class FinancialHealthProfile(BaseModel):
    """
    Assessment of debtor's current liquidity and benefit streams.
    Captured during SubStage_3_2_FinancialProfiling.
    """

    model_config = ConfigDict(graph_id_fields=["profileRFD"])

    profile_rfd: RFDEnum = Field(
        ...,
        description="Root cause of delinquency identified from this financial profile.",
        json_schema_extra={"edge_label": "profileRFD"},
    )
    is_receiving_ui_benefits: Optional[bool] = Field(
        None,
        description="Whether the debtor is receiving unemployment insurance benefits.",
    )
    ui_benefit_amount: Optional[Decimal] = Field(
        None,
        description="Monthly UI benefit amount (abstracted; do not capture exact values).",
    )
    severance_amount: Optional[Decimal] = Field(
        None,
        description="Severance payment received (abstracted; do not capture exact values).",
    )


class ComplianceArtifact(BaseModel):
    """
    Mandatory regulatory or audit proof logged during a call.
    References one of the predefined compliance requirement individuals.
    """

    model_config = ConfigDict(graph_id_fields=["artifact_id"])

    artifact_id: ComplianceArtifactEnum = Field(
        ..., description="Which compliance checkpoint was satisfied."
    )
    label: Optional[str] = Field(
        None, description="Human-readable label for the compliance artifact."
    )


# =============================================================================
# 4. State Machine Models — SOP topology
# =============================================================================


class MasterSOPStage(BaseModel):
    """
    One of the 7 core stages in the master collection pipeline.
    Forms a linear progression via nextMasterStage.
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


class SubStage(BaseModel):
    """
    Specialized dialogue sub-states within the M1-Unemployment branch.
    Flows via nextSubStage, then rejoins the master pipeline.
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


# =============================================================================
# 5. Outcome Models — discriminated union
# =============================================================================


class PromiseToPay(BaseModel):
    """Binding commitment to pay a specified amount on a future date."""

    outcome_type: Literal["PromiseToPay"] = "PromiseToPay"
    ptp_amount: Optional[Decimal] = Field(
        None,
        description="Promised payment amount (abstract; avoid capturing raw values).",
    )
    ptp_due_date: Optional[date] = Field(
        None, description="Date the payment was promised for."
    )
    is_auto_pay_configured: Optional[bool] = Field(
        None, description="Whether auto-pay was set up as part of the PTP."
    )


class ForbearanceAgreement(BaseModel):
    """Temporary hold or restructuring granted due to unemployment hardship."""

    outcome_type: Literal["ForbearanceAgreement"] = "ForbearanceAgreement"
    forbearance_duration_days: Optional[int] = Field(
        None, description="Duration of the forbearance period in days."
    )
    restructured_terms: Optional[str] = Field(
        None, description="Brief description of restructured terms (abstracted)."
    )


class MicroCommitment(BaseModel):
    """Non-monetary agreement, e.g., agreeing to provide a job status update."""

    outcome_type: Literal["MicroCommitment"] = "MicroCommitment"
    commitment_description: Optional[str] = Field(
        None, description="What the debtor agreed to do (non-monetary)."
    )
    follow_up_date: Optional[date] = Field(
        None, description="When the micro-commitment is expected to be fulfilled."
    )


CallOutcome = Annotated[
    Union[PromiseToPay, ForbearanceAgreement, MicroCommitment],
    Field(discriminator="outcome_type"),
]


# =============================================================================
# 6. Core Entity Models
# =============================================================================


class DebtAccount(BaseModel):
    """The financial obligation record subject to collection."""

    model_config = ConfigDict(graph_id_fields=["account_id"])

    account_id: str = Field(
        ...,
        description="Anonymous account identifier (use placeholder, not real account number).",
    )
    account_number: Optional[str] = Field(
        None, description="Account number — **avoid populating with real values**."
    )
    outstanding_balance: Optional[Decimal] = Field(
        None,
        description="Outstanding balance — **abstract or leave null for PII safety**.",
    )
    days_past_due: Optional[int] = Field(
        None, description="Number of days the account is past due."
    )
    delinquency_stage: Literal["M1"] = Field(
        "M1", description="Delinquency bucket; expected 'M1' for 1-30 DPD."
    )


class Debtor(BaseModel):
    """An individual owing an outstanding debt in early-stage (M1) delinquency."""

    model_config = ConfigDict(graph_id_fields=["debtor_id"])

    debtor_id: str = Field(
        ..., description="Anonymous debtor identifier (e.g., 'Debtor_Anonymous')."
    )
    label: Optional[str] = Field(None, description="Optional label — avoid real names.")
    has_account: list[DebtAccount] = Field(
        default_factory=list,
        description="Accounts held by this debtor.",
        json_schema_extra={"edge_label": "hasAccount"},
    )
    has_financial_profile: Optional[FinancialHealthProfile] = Edge(
        label="hasFinancialProfile",
        description="The debtor's financial health profile.",
    )
    experiences_hardship: Optional[RFDEnum] = Edge(
        label="experiencesHardship",
        description="Hardship reason experienced by the debtor.",
    )

    @field_validator("debtor_id")
    @classmethod
    def anonymize_id(cls, v: str) -> str:
        """Ensure the ID is treated as anonymous."""
        return v.strip()


class CollectionAgent(BaseModel):
    """Human collector or automated dialogue bot conducting the interaction."""

    model_config = ConfigDict(graph_id_fields=["agent_id"])

    agent_id: str = Field(
        ..., description="Anonymous agent identifier (e.g., 'Agent_Anonymous')."
    )
    label: Optional[str] = Field(None, description="Optional label.")


class Call(BaseModel):
    """
    A specific telecommunication dialogue session between an Agent and a Debtor.
    The central execution entity linking all other nodes.
    """

    model_config = ConfigDict(graph_id_fields=["call_id"])

    call_id: str = Field(..., description="Unique anonymous call identifier.")

    # --- Actor & Account Links ---
    conducted_by_agent: Optional[CollectionAgent] = Edge(
        label="conductedByAgent",
        description="The agent who conducted this call.",
    )
    has_debtor_participant: Optional[Debtor] = Edge(
        label="hasDebtorParticipant",
        description="The debtor who participated in this call.",
    )
    concerns_account: Optional[DebtAccount] = Edge(
        label="concernsAccount",
        description="The debt account this call concerns.",
    )

    # --- Financial & Hardship Links ---
    evaluated_financials: Optional[FinancialHealthProfile] = Edge(
        label="evaluatedFinancials",
        description="Financial health profile evaluated during this call.",
    )
    detected_rfd: Optional[RFDEnum] = Edge(
        label="detectedRFD",
        description="Reason for Delinquency detected during this call.",
    )

    # --- SOP Progression ---
    current_master_stage: Optional[MasterSOPStageEnum] = Edge(
        label="currentMasterStage",
        description="The real-time master SOP stage of the ongoing call.",
    )
    current_sub_stage: Optional[SubStageEnum] = Edge(
        label="currentSubStage",
        description="The real-time sub-stage (if in M1-Unemployment branch).",
    )

    # --- Compliance ---
    logged_compliance: list[ComplianceArtifactEnum] = Field(
        default_factory=list,
        description="Compliance checkpoints satisfied during this call.",
        json_schema_extra={"edge_label": "loggedCompliance"},
    )

    # --- Outcome ---
    yielded_outcome: Optional[CallOutcome] = Edge(
        label="yieldedOutcome",
        description="Formal resolution or commitment resulting from this call.",
    )

    # --- Call Meta ---
    call_start_time: Optional[datetime] = Field(
        None, description="Call start timestamp."
    )
    call_end_time: Optional[datetime] = Field(None, description="Call end timestamp.")
    is_verification_passed: Optional[bool] = Field(
        None, description="Whether right-party verification passed."
    )


# =============================================================================
# 7. Root Extraction Model
# =============================================================================


class DebtCollectionExtraction(BaseModel):
    """
    Root model for LLM extraction from a single M1-Unemployment debt collection call.
    Captures the full connected graph: Call, Debtor, Account, Financial Profile,
    SOP stages, compliance artifacts, and outcome.
    """

    call: Call = Field(
        ..., description="The primary call entity extracted from the dialogue."
    )
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
