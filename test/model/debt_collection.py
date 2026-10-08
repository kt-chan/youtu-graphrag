"""流程编排知识图谱模型。设计要点：
- 节点类型与关系通过schema.json声明
- 枚举定义节点状态与语义
- 模型封装节点属性与连接关系
- 根提取模型聚合全部实例
"""

from __future__ import annotations
from typing import Optional, List, Dict
from pydantic import BaseModel, ConfigDict, Field
from .graph_base import Edge, ListEdge, GraphNodeEnum, NodeModel

# === Enums ===

class MasterStageEnum(GraphNodeEnum):
    """主阶段节点。"""
    MASTER_STAGE_INIT = "MasterStage_MasterStageInit"
    MASTER_STAGE_EXEC = "MasterStage_MasterStageExec"
    MASTER_STAGE_VERIFICATION = "MasterStage_MasterStageVerification"
    MASTER_STAGE_COMPLETE = "MasterStage_MasterStageComplete"

    @classmethod
    def member_descriptions(cls) -> dict[str, str]:
        return {
            "MasterStage_MasterStageInit": "初始化主阶段：定义阶段边界与前置条件。",
            "MasterStage_MasterStageExec": "执行主阶段：包含核心业务流程与决策点。",
            "MasterStage_MasterStageVerification": "验证主阶段：校验执行结果与合规性。",
            "MasterStage_MasterStageComplete": "完成主阶段：记录阶段状态与产出。",
        }

class SubflowCategoryEnum(GraphNodeEnum):
    """子流程分类节点。"""
    CATEGORY_INIT = "SubflowCategory_CategoryInit"
    CATEGORY_NORMAL = "SubflowCategory_CategoryNormal"
    CATEGORY_ERROR = "SubflowCategory_CategoryError"

    @classmethod
    def member_descriptions(cls) -> dict[str, str]:
        return {
            "SubflowCategory_CategoryInit": "初始化分类：准备子流程启动环境。",
            "SubflowCategory_CategoryNormal": "常规分类：标准执行路径。",
            "SubflowCategory_CategoryError": "异常分类：处理流程中断与异常。",
        }

class SubStepEnum(GraphNodeEnum):
    """子步骤节点。"""
    STEP_INIT = "SubStep_StepInit"
    STEP_EXEC = "SubStep_StepExec"
    STEP_VERIFICATION = "SubStep_StepVerification"
    STEP_COMPLETE = "SubStep_StepComplete"

    @classmethod
    def member_descriptions(cls) -> dict[str, str]:
        return {
            "SubStep_StepInit": "初始化子步骤：准备步骤执行资源。",
            "SubStep_StepExec": "执行子步骤：执行具体操作与计算。",
            "SubStep_StepVerification": "验证子步骤：检查步骤执行结果。",
            "SubStep_StepComplete": "完成子步骤：记录步骤状态与产出。",
        }

class ComplianceArtifactEnum(GraphNodeEnum):
    """合规文档节点。"""
    ARTIFACT_INIT = "ComplianceArtifact_ArtifactInit"
    ARTIFACT_GENERATE = "ComplianceArtifact_ArtifactGenerate"
    ARTIFACT_STORE = "ComplianceArtifact_ArtifactStore"

    @classmethod
    def member_descriptions(cls) -> dict[str, str]:
        return {
            "ComplianceArtifact_ArtifactInit": "初始化文档：准备文档模板与元数据。",
            "ComplianceArtifact_ArtifactGenerate": "生成文档：根据模板填充内容。",
            "ComplianceArtifact_ArtifactStore": "存储文档：归档已生成的合规文件。",
        }

class ObjectionEnum(GraphNodeEnum):
    """异议节点。"""
    OBJECTION_INIT = "Objection_ObjectionInit"
    OBJECTION_RECORD = "Objection_ObjectionRecord"
    OBJECTION_REVIEW = "Objection_ObjectionReview"
    OBJECTION_RESOLVE = "Objection_ObjectionResolve"

    @classmethod
    def member_descriptions(cls) -> dict[str, str]:
        return {
            "Objection_ObjectionInit": "初始化异议：准备异议处理流程。",
            "Objection_ObjectionRecord": "记录异议：收集与登记异议信息。",
            "Objection_ObjectionReview": "审核异议：评估异议有效性。",
            "Objection_ObjectionResolve": "解决异议：制定并执行异议处理方案。",
        }

# === Canonical Graph Node Models ===

class MasterStage(NodeModel):
    """主阶段节点。主阶段按时间线性推进，包含多个子流程与合规检查点。"""
    model_config = ConfigDict(graph_id_fields=["stage_id"])

    stage_id: MasterStageEnum = Field(..., description="主阶段唯一标识。")
    next_master_stage: Optional[MasterStageEnum] = Edge(
        label="nextMasterStage", default=None,
        description="指向下一阶段的主阶段连接，表示流程线性推进。",
    )
    branches_to_category: Optional[SubflowCategoryEnum] = Edge(
        label="branchesToCategory", default=None,
        description="分支指向的子流程分类，表示流程路径选择。",
    )
    requires_compliance: Optional[ComplianceArtifactEnum] = Edge(
        label="requiresCompliance", default=None,
        description="需要关联的合规文档，表示合规性要求。",
    )
    yields_outcome: Optional[MasterStageEnum] = Edge(
        label="yieldsOutcome", default=None,
        description="产生的阶段结果，表示阶段产出。",
    )
    parent_master_stage: Optional[MasterStageEnum] = Edge(
        label="parentMasterStage", default=None,
        description="前置主阶段，表示流程层级关系。",
    )
    initial_sub_step: Optional[SubStepEnum] = Edge(
        label="initialSubStep", default=None,
        description="初始子步骤，表示阶段入口操作。",
    )
    rejoins_master_stage: Optional[MasterStageEnum] = Edge(
        label="rejoinsMasterStage", default=None,
        description="重新连接的主阶段，表示流程合并点。",
    )
    satisfied_in_stage: Optional[MasterStageEnum] = Edge(
        label="satisfiedInStage", default=None,
        description="满足的阶段条件，表示流程触发条件。",
    )
    stage_name: str = Field(..., description="主阶段名称。")
    stage_class_: str = Field(alias="class", description="主阶段分类。")
    extracted_info: list[str] = Field(default_factory=list, description="阶段产生的自由形式事实信息。")

class SubflowCategory(NodeModel):
    """子流程分类节点。分类定义子流程的执行模式与约束条件。"""
    model_config = ConfigDict(graph_id_fields=["category_id"])

    category_id: SubflowCategoryEnum = Field(..., description="子流程分类唯一标识。")
    parent_category: Optional[SubflowCategoryEnum] = Edge(
        label="parentCategory", default=None,
        description="前置子流程分类，表示分类层级关系。",
    )
    next_master_stage: Optional[MasterStageEnum] = Edge(
        label="nextMasterStage", default=None,
        description="关联的主阶段，表示分类所属流程。",
    )
    name: str = Field(..., description="分类名称。")
    class_: str = Field(alias="class", description="分类类型。")
    extracted_info: list[str] = Field(default_factory=list, description="分类产生的自由形式事实信息。")

class SubStep(NodeModel):
    """子步骤节点。子步骤是可复用的操作单元，包含具体执行动作与验证点。"""
    model_config = ConfigDict(graph_id_fields=["sub_step_id"])

    sub_step_id: SubStepEnum = Field(..., description="子步骤唯一标识。")
    next_sub_step: Optional[SubStepEnum] = Edge(
        label="nextSubStep", default=None,
        description="指向下一子步骤的连接，表示步骤序列。",
    )
    parent_category: Optional[SubflowCategoryEnum] = Edge(
        label="parentCategory", default=None,
        description="所属的子流程分类，表示步骤归属。",
    )
    triggered_in_step: Optional[SubStepEnum] = Edge(
        label="triggeredInStep", default=None,
        description="触发该步骤的子步骤，表示步骤依赖。",
    )
    parent_master_stage: Optional[MasterStageEnum] = Edge(
        label="parentMasterStage", default=None,
        description="所属的主阶段，表示步骤层级关系。",
    )
    name: str = Field(..., description="步骤名称。")
    class_: str = Field(alias="class", description="步骤类型。")
    extracted_info: list[str] = Field(default_factory=list, description="步骤产生的自由形式事实信息。")

class ComplianceArtifact(NodeModel):
    """合规文档节点。文档记录流程执行过程中的合规性证据与记录。"""
    model_config = ConfigDict(graph_id_fields=["artifact_id"])

    artifact_id: ComplianceArtifactEnum = Field(..., description="合规文档唯一标识。")
    yields_outcome: Optional[MasterStageEnum] = Edge(
        label="yieldsOutcome", default=None,
        description="产生的阶段结果，表示文档关联。",
    )
    parent_master_stage: Optional[MasterStageEnum] = Edge(
        label="parentMasterStage", default=None,
        description="所属的主阶段，表示文档归属。",
    )
    name: str = Field(..., description="文档名称。")
    class_: str = Field(alias="class", description="文档类型。")
    extracted_info: list[str] = Field(default_factory=list, description="文档产生的自由形式事实信息。")

class Objection(NodeModel):
    """异议节点。记录与处理流程执行过程中的异议与申诉信息。"""
    model_config = ConfigDict(graph_id_fields=["objection_id"])

    objection_id: ObjectionEnum = Field(..., description="异议唯一标识。")
    satisfied_in_stage: Optional[MasterStageEnum] = Edge(
        label="satisfiedInStage", default=None,
        description="满足的阶段条件，表示异议关联。",
    )
    parent_category: Optional[SubflowCategoryEnum] = Edge(
        label="parentCategory", default=None,
        description="所属的子流程分类，表示异议归属。",
    )
    name: str = Field(..., description="异议名称。")
    class_: str = Field(alias="class", description="异议类型。")
    extracted_info: list[str] = Field(default_factory=list, description="异议产生的自由形式事实信息。")

# === Root Extraction Model ===

class DataExtraction(BaseModel):
    """流程编排知识图谱根提取模型。生成顺序建议：MasterStage -> SubflowCategory -> SubStep -> ComplianceArtifact -> Objection
    连通性硬约束：所有节点ID必须有效，关系连接必须符合业务逻辑，不存在悬空ID或违反基数约束的情况。
    """
    master_stages: List[MasterStage] = Field(default_factory=list, description="主阶段实例列表。")
    subflow_categories: List[SubflowCategory] = Field(default_factory=list, description="子流程分类实例列表。")
    sub_steps: List[SubStep] = Field(default_factory=list, description="子步骤实例列表。")
    compliance_artifacts: List[ComplianceArtifact] = Field(default_factory=list, description="合规文档实例列表。")
    objections: List[Objection] = Field(default_factory=list, description="异议实例列表。")
