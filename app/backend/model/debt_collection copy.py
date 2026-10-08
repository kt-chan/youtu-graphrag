# app/backend/model/debt_collection.py
"""
M1 Debt Recovery SOP Ontology — Fully Connected Graph Models (Pydantic v2)

九步线性主流程（S1 → S9）。

设计要点：
- `GraphNodeEnum`：枚举自身声明"是否为图节点 + 节点前缀 + 中文决策说明"。
- 枚举名去掉 `Enum` 后缀后与对应 Pydantic 模型类名一致，保证 enum 引用
  与模型实例解析出同一 node_id。
- 所有 enum 成员附 `member_descriptions()`，序列化时注入 JSON Schema 的
  `x-enum-descriptions`。
- 所有字段 `description` 均为中文，随 Schema 进入 prompt。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# Edge Helper
# =============================================================================

def Edge(label: str, default: Any = None, required: bool = False, **kwargs):
    if required:
        return Field(..., json_schema_extra={"edge_label": label}, **kwargs)
    return Field(default, json_schema_extra={"edge_label": label}, **kwargs)


# =============================================================================
# Base class for node-producing enums
# =============================================================================

class GraphNodeEnum(str, Enum):
    """枚举成员作为知识图谱节点。节点 ID 为 `{node_prefix()}__{member_value}`。"""

    @classmethod
    def node_prefix(cls) -> str:
        return cls.__name__.removesuffix("Enum")

    @classmethod
    def member_descriptions(cls) -> dict[str, str]:
        return {}

    @classmethod
    def description_for(cls, value: str) -> Optional[str]:
        for sub in GraphNodeEnum.__subclasses__():
            desc = sub.member_descriptions().get(value)
            if desc:
                return desc
        return None


# =============================================================================
# Enums
# =============================================================================

class MasterStageEnum(GraphNodeEnum):
    """主流程阶段。九步线性顺序 S1 → S9。

    约束：
    - 通话中出现的阶段按顺序排列且不重复（允许跳过）。
    - 仅最后一个实际到达的阶段 `next_master_stage` 为 null。
    - 分支进入子流程固定从 S3_NeedsDiscovery 发生。
    - Outcome_* 仅挂载在 S5_ArrangementNegotiation。
    """

    S1_RPC_VERIFICATION          = "S1_RPCVerification"
    S2_BALANCE_DISCLOSURE        = "S2_BalanceDisclosure"
    S3_NEEDS_DISCOVERY           = "S3_NeedsDiscovery"
    S4_URGENCY_CREATION          = "S4_UrgencyCreation"
    S5_ARRANGEMENT_NEGOTIATION   = "S5_ArrangementNegotiation"
    S6_PTP_COMMITMENT            = "S6_PTPCommitment"
    S7_SKIP_TRACING              = "S7_SkipTracing"
    S8_PROFESSIONAL_INTERRUPTION = "S8_ProfessionalInterruption"
    S9_CALL_CLOSE                = "S9_CallClose"

    @classmethod
    def member_descriptions(cls) -> dict[str, str]:
        return {
            "S1_RPCVerification": (
                "Right-Party Contact & Verification：确认与账户持有人本人通话，"
                "核实全名、地址、出生日期或SSN后四位等身份信息，完成隐私合规披露。"
                "所有外呼和内呼均须执行。"
            ),
            "S2_BalanceDisclosure": (
                "Balance Disclosure：以中性、清晰的措辞披露欠款金额、"
                "相关账户及原始债权人信息。避免使用可能引发争议的表述。"
            ),
            "S3_NeedsDiscovery": (
                "Needs Discovery / Fact-Finding：通过开放式提问了解客户逾期原因、"
                "当前财务状况与还款意愿。此处是判定困难类型、进入子流程的分支点。"
            ),
            "S4_UrgencyCreation": (
                "Urgency Creation & Consequences：在FDCPA/Reg F等法规框架内，"
                "说明逾期可能产生的后果，建立合理紧迫感，不使用威胁或误导性语言。"
            ),
            "S5_ArrangementNegotiation": (
                "Payment Arrangement Negotiation：基于客户实际偿付能力，"
                "协商一次性付清、分期还款计划或和解方案。是唯一允许挂载 Outcome_* 的阶段。"
            ),
            "S6_PTPCommitment": (
                "Promise-to-Pay (PTP) Commitment：将口头意向转为明确的PTP，"
                "确认还款金额、日期和方式，并复述确认以消除歧义。"
            ),
            "S7_SkipTracing": (
                "Skip Tracing / Contact Update：确认或更新客户的备用联系方式，"
                "降低后续失联风险，确保后续跟进渠道畅通。仅接在 S6 之后。"
            ),
            "S8_ProfessionalInterruption": (
                "Professional Call Interruption：客户不便通话时，"
                "礼貌中断并约定回拨时间，保留后续沟通空间。协商未成时的收尾路径。完成后默认进入 S9。"
            ),
            "S9_CallClose": (
                "Call Close & Documentation：复述确认达成的协议，"
                "提供催收员姓名及回拨号码，记录通话结果、PTP及后续跟进事项。"
            ),
        }


class ComplianceArtifactEnum(GraphNodeEnum):
    """合规检查点。每项需在通话中被实际执行，并在 satisfied_in_stage 指向满足位置。"""

    COMP_VERIFICATION     = "Compliance_Verification"
    COMP_DISCLOSURE       = "Compliance_Disclosure"
    COMP_EXPLICIT_CONSENT = "Compliance_ExplicitConsent"
    COMP_HARDSHIP_TERMS   = "Compliance_HardshipTerms"

    @classmethod
    def member_descriptions(cls) -> dict[str, str]:
        return {
            "Compliance_Verification":     "身份核实：确认对方为持卡人本人（S1 执行）。",
            "Compliance_Disclosure":       "信息披露：告知欠款金额、逾期天数与后续影响（S1/S2 执行）。",
            "Compliance_ExplicitConsent":  "明确授权：获取客户对沟通或方案的明确同意。",
            "Compliance_HardshipTerms":    "困难政策告知：告知减免/延期的条款边界与所需材料。",
        }


class ObjectionEnum(GraphNodeEnum):
    """客户异议分类。"""

    OBJ_FINANCIAL_HARDSHIP = "Objection_Financial_Hardship"
    OBJ_NO_INCOME          = "Objection_No_Income"
    OBJ_WAITING_BENEFITS   = "Objection_Waiting_Benefits"
    OBJ_REFUSAL            = "Objection_Refusal"

    @classmethod
    def member_descriptions(cls) -> dict[str, str]:
        return {
            "Objection_Financial_Hardship": "财务困难异议：以经济困难为由拒绝还款。",
            "Objection_No_Income":          "无收入异议：表示当前没有任何收入来源。",
            "Objection_Waiting_Benefits":   "等待补助：在等失业金/救助到账。",
            "Objection_Refusal":            "拒绝异议：明确拒绝沟通或还款。",
        }


class OutcomeEnum(GraphNodeEnum):
    """S5_ArrangementNegotiation 阶段产出的结果类型。仅可挂载在 S5 上。"""

    OUTCOME_PTP              = "Outcome_PTP"
    OUTCOME_TOKEN_PAYMENT    = "Outcome_TokenPayment"
    OUTCOME_FORBEARANCE      = "Outcome_Forbearance"
    OUTCOME_MICRO_COMMITMENT = "Outcome_MicroCommitment"

    @classmethod
    def member_descriptions(cls) -> dict[str, str]:
        return {
            "Outcome_PTP":              "达成 PTP：客户承诺具体日期与金额的全额/分期还款。",
            "Outcome_TokenPayment":     "部分还款：客户先还一小部分，余款另约。",
            "Outcome_Forbearance":      "延期：同意暂停催收或延期还款。",
            "Outcome_MicroCommitment":  "微承诺：客户做出小额/短期的可执行动作，但未形成完整 PTP。",
        }


class SubflowCategoryEnum(GraphNodeEnum):
    """子流程分类。从 S3_NeedsDiscovery 进入，必须回挂到某个主阶段。

    与模型 `SubflowCategory` 构成 1:1：枚举成员即该模型实例的 `category_id`。
    """

    FLOW_UNEMPLOYMENT = "Flow_Unemployment"
    FLOW_OTHER        = "Flow_Other"

    @classmethod
    def member_descriptions(cls) -> dict[str, str]:
        return {
            "Flow_Unemployment": "失业子流程：客户明确因失业导致逾期的专属分支。",
            "Flow_Other":        "其他困难子流程：非失业类困难的专属分支。",
        }


class SubStepEnum(GraphNodeEnum):
    """子步骤：进入子流程后的具体对话阶段。

    约束：
    - 每个子步骤属于且仅属于一个子流程分类，严禁跨分类跳转。
    - Flow_Unemployment 路径：Empathy_Verify → Alt_Resources
      → Hardship_Pitch → Micro_Commitment。
    - Flow_Other 路径：Root_Cause → Impact_Assess → Custom_Pitch。
    """

    # Flow_Unemployment
    SUB_EMPATHY_VERIFY  = "SubStep_Empathy_Verify"
    SUB_ALT_RESOURCES   = "SubStep_Alt_Resources"
    SUB_HARDSHIP_PITCH  = "SubStep_Hardship_Pitch"
    SUB_MICRO_COMMIT    = "SubStep_Micro_Commitment"
    # Flow_Other
    SUB_ROOT_CAUSE      = "SubStep_Root_Cause"
    SUB_IMPACT_ASSESS   = "SubStep_Impact_Assess"
    SUB_CUSTOM_PITCH    = "SubStep_Custom_Pitch"

    @classmethod
    def member_descriptions(cls) -> dict[str, str]:
        return {
            "SubStep_Empathy_Verify":   "共情确认：先共情，再核实失业细节（离职时间、求职状态、家庭负担）。客户首次表露失业时使用。",
            "SubStep_Alt_Resources":    "替代资源：引导客户挖掘可动用资金（家人、失业保险、兼职、其他账户），暂不谈还款。",
            "SubStep_Hardship_Pitch":   "困难话术：结合困难提出个性化方案（减免、延期、分期），说明政策边界与所需材料。",
            "SubStep_Micro_Commitment": "微承诺：引导客户做出小额、具体、可执行的承诺（如明天先还515元），并复述确认。",
            "SubStep_Root_Cause":       "根因挖掘：追问逾期真实原因（非失业类：收入下降、医疗、家庭变故等）。客户理由模糊时使用。",
            "SubStep_Impact_Assess":    "影响评估：说明逾期对征信、额度、后续贷款的后果，结合客户在意场景提示风险。",
            "SubStep_Custom_Pitch":     "定制话术：基于根因与客户在意点给出针对性建议与政策组合，避免通用话术。",
        }


# =============================================================================
# Canonical Graph Node Models
# =============================================================================

class ComplianceArtifact(BaseModel):
    """合规检查点节点。"""

    model_config = ConfigDict(graph_id_fields=["artifact_id"])

    artifact_id: ComplianceArtifactEnum = Field(..., description="合规检查点类型。")
    satisfied_in_stage: Optional[Union[MasterStageEnum]] = Edge(
        label="satisfiedInStage",
        default=None,
        description="该合规项被满足的阶段或子步骤，必须存在于 master_stages 中。",
    )


class MasterStage(BaseModel):
    """主流程节点。每个通话按时间顺序出现在 master_stages 列表中，且不重复。"""

    model_config = ConfigDict(graph_id_fields=["stage_id"])

    stage_id: MasterStageEnum = Field(..., description="主阶段标识。")

    next_master_stage: Optional[MasterStageEnum] = Edge(
        label="nextMasterStage",
        default=None,
        description="本阶段之后的下一主阶段（主阶段值，非子流程分类）。仅通话到达的最后一个阶段为 null。",
    )
    branches_to_category: Optional[SubflowCategoryEnum] = Edge(
        label="branchesToCategory",
        default=None,
        description="从 S3_NeedsDiscovery 分支出的子流程分类（子流程值，非主阶段）。仅 S3 允许非空。",
    )
    required_compliance: list[ComplianceArtifactEnum] = Field(
        default_factory=list,
        description=(
            "本阶段需要满足的合规项列表。"
            "元素为 ComplianceArtifactEnum 字符串，如 'Compliance_Verification'，不是对象。"
        ),
        json_schema_extra={"edge_label": "requiresCompliance"},
    )
    stage_outcomes: list[OutcomeEnum] = Field(
        default_factory=list,
        description="本阶段产出的结果类型。仅当 stage_id 为 'S5_ArrangementNegotiation' 时允许非空。",
        json_schema_extra={"edge_label": "yieldsOutcome"},
    )


class SubflowCategory(BaseModel):
    """子流程分类节点。父阶段固定为 S3_NeedsDiscovery。"""

    model_config = ConfigDict(graph_id_fields=["category_id"])

    category_id: SubflowCategoryEnum = Field(..., description="子流程分类标识。")
    parent_master_stage: Literal[MasterStageEnum.S3_NEEDS_DISCOVERY] = Field(
        default=MasterStageEnum.S3_NEEDS_DISCOVERY,
        description="父主阶段，固定为 S3_NeedsDiscovery。",
        json_schema_extra={"edge_label": "parentMasterStage"},
    )
    initial_sub_step: Optional[SubStepEnum] = Edge(
        label="initialSubStep",
        default=None,
        description="本子流程的第一个子步骤。未产生子步骤时为 null。",
    )


class SubStep(BaseModel):
    """子步骤节点。parent_category 与 next_sub_step 必须同属一个分类。"""

    model_config = ConfigDict(graph_id_fields=["sub_step_id"])

    sub_step_id: SubStepEnum = Field(..., description="子步骤标识。")
    parent_category: SubflowCategoryEnum = Edge(
        label="parentCategory",
        required=True,
        description="所属子流程分类（Flow_Unemployment 或 Flow_Other）。",
    )
    extracted_info: list[str] = Field(
        default_factory=list,
        description="本步骤中抽取到的关键事实（如“失业”“承诺明天还515元”）。",
    )
    next_sub_step: Optional[SubStepEnum] = Edge(
        label="nextSubStep",
        default=None,
        description="同分类下的下一子步骤。无后续时为 null。",
    )
    rejoins_master_stage: Optional[
        Literal[
            MasterStageEnum.S4_URGENCY_CREATION,
            MasterStageEnum.S5_ARRANGEMENT_NEGOTIATION,
            MasterStageEnum.S6_PTP_COMMITMENT,
            MasterStageEnum.S7_SKIP_TRACING,
            MasterStageEnum.S8_PROFESSIONAL_INTERRUPTION,
            MasterStageEnum.S9_CALL_CLOSE,
        ]
    ] = Edge(
        label="rejoinsMasterStage",
        default=None,
        description=(
            "子流程回挂的主阶段，只能是 S4 及之后的阶段"
            "（S4_UrgencyCreation / S5_ArrangementNegotiation / S6_PTPCommitment / "
            "S7_SkipTracing / S8_ProfessionalInterruption / S9_CallClose）。"
        ),
    )


class Objection(BaseModel):
    """客户异议节点。triggered_in_step 必须存在于 master_stages 中。"""

    model_config = ConfigDict(graph_id_fields=["objection_id"])

    objection_id: ObjectionEnum = Field(..., description="异议类型。")
    triggered_in_step: Optional[MasterStageEnum] = Edge(
        label="triggeredInStep",
        default=None,
        description="异议被触发的主阶段，必须存在于 master_stages 中。",
    )


# =============================================================================
# Root Extraction Model  - Default to DataExtraction
# =============================================================================

class DataExtraction(BaseModel):
    """通话结构化抽取的根对象。

    生成顺序建议：
      1. 先按时间列出 master_stages（唯一、线性、无重复）。
      2. 若进入子流程，填写 subflow_categories 与 sub_steps。
      3. 最后补齐 compliance_artifacts 与 objections_raised。

    连通性硬约束：
      - 引用的节点 ID 必须在本对象内被显式定义，严禁悬空。
      - Outcome_* 仅可挂在 S5_ArrangementNegotiation.stage_outcomes 上。
      - SubStep.rejoins_master_stage 只能为 S4 及之后的阶段。
      - 未进入子流程时，subflow_categories 与 sub_steps 必须为 []。
    """

    master_stages: list[MasterStage] = Field(
        default_factory=list, description="按时间顺序出现的主阶段列表。"
    )
    subflow_categories: list[SubflowCategory] = Field(
        default_factory=list, description="本次通话进入的子流程分类；未进入则为 []。"
    )
    sub_steps: list[SubStep] = Field(
        default_factory=list, description="子流程内的子步骤列表；未进入则为 []。"
    )
    compliance_artifacts: list[ComplianceArtifact] = Field(
        default_factory=list, description="通话中实际满足的合规检查点。"
    )
    objections_raised: list[Objection] = Field(
        default_factory=list, description="通话中客户提出并触发处理的异议。"
    )