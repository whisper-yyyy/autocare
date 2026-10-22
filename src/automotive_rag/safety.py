"""Deterministic automotive safety intent classification.

Safety decisions run before retrieval so an empty or incomplete knowledge base
cannot suppress urgent guidance.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SafetyDecision:
    mode: str
    response: str
    notice: str
    refused: bool


SAFETY_COMPONENTS = (
    "安全气囊",
    "气囊",
    "安全带",
    "abs",
    "制动",
    "刹车",
    "转向",
    "胎压",
    "传感器",
    "故障灯",
    "警告灯",
    "报警",
)

BYPASS_ACTIONS = (
    "绕过",
    "屏蔽",
    "短接",
    "拔掉",
    "拔出",
    "断开",
    "拆掉",
    "拆除",
    "关闭",
    "关掉",
    "停用",
    "禁用",
    "欺骗",
    "消除",
    "清除",
    "刷掉",
    "不亮",
)

CONTINUE_DRIVING_TERMS = (
    "继续开",
    "继续驾驶",
    "开到修理厂",
    "开去修理厂",
    "还能开",
    "能继续开",
    "带病行驶",
)

HIGH_RISK_GROUPS = {
    "fire": ("着火", "起火", "烧起来", "燃烧", "冒烟", "焦糊味"),
    "brake": (
        "制动失灵",
        "刹车失灵",
        "刹不住",
        "制动踏板变软",
        "刹车踏板变软",
        "制动力下降",
        "制动距离异常",
    ),
    "steering": ("转向失灵", "方向盘失控", "方向失控"),
    "tire": ("爆胎", "轮胎爆裂"),
    "overheat": ("水温过高", "发动机过热", "高温报警"),
}

EXACT_DANGEROUS_TERMS = (
    "绕过气囊",
    "关闭气囊",
    "短接安全",
    "屏蔽故障报警",
    "拆除安全带",
)


def _risk_group(question: str) -> Optional[str]:
    for group, terms in HIGH_RISK_GROUPS.items():
        for term in terms:
            if term == "着火" and ("不着火" in question or "打不着火" in question):
                continue
            if term in question:
                return group
    return None


def _emergency_response(group: str) -> str:
    if group == "fire":
        return (
            "这属于高风险情况。请在确保人员安全的前提下立即停车、熄火并撤离到安全区域，"
            "不要继续驾驶或自行拆卸车辆；请联系消防、道路救援或专业维修人员。"
        )
    return (
        "这属于高风险行车症状。请在确保安全的前提下尽快停车并停止继续驾驶，"
        "开启危险警示灯，联系道路救援或专业维修人员检查。"
    )


def classify_safety(question: str) -> Optional[SafetyDecision]:
    """Classify bypass attempts and urgent symptoms using conservative rules."""

    normalized = question.lower().strip()
    if not normalized:
        return None

    risk_group = _risk_group(normalized)
    component_action = (
        any(component in normalized for component in SAFETY_COMPONENTS)
        and any(action in normalized for action in BYPASS_ACTIONS)
    )
    dangerous_continue = risk_group is not None and any(
        term in normalized for term in CONTINUE_DRIVING_TERMS
    )
    if (
        any(term in normalized for term in EXACT_DANGEROUS_TERMS)
        or component_action
        or dangerous_continue
    ):
        return SafetyDecision(
            mode="safety-refusal",
            response=(
                "我不能提供绕过、停用车辆安全系统或在高风险故障下继续驾驶的方法。"
                "请停止相关操作，在确保人员安全的前提下停车，并联系道路救援或授权维修机构。"
            ),
            notice="该请求涉及规避车辆安全装置或在高风险状态下继续行驶。",
            refused=True,
        )

    if risk_group is not None:
        return SafetyDecision(
            mode="safety-guidance",
            response=_emergency_response(risk_group),
            notice="问题包含潜在高风险症状；安全处置优先于知识库检索结果。",
            refused=False,
        )
    return None
