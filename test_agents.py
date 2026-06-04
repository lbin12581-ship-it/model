import unittest

from summary_system.agents import AgentOrchestrator
from summary_system.models import InputDocument


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.orchestrator = AgentOrchestrator(llm=False)

    def test_meeting_summary_extracts_task(self):
        text = "张三：今天会议讨论项目上线。决定下周发布。李四负责完成测试，截止6月1日。"
        result = self.orchestrator.run(InputDocument(title="测试会议", raw_text=text, mode="auto"))
        self.assertEqual(result.scene.scene_type, "meeting")
        self.assertIn("待办事项", result.content)
        self.assertTrue(result.content["待办事项"])

    def test_classroom_summary_detects_formula(self):
        text = "老师：本节课讲二次函数。重点是公式 y=ax^2+bx+c，同学们考试要注意定义域。"
        result = self.orchestrator.run(InputDocument(title="课堂", raw_text=text, mode="auto"))
        self.assertEqual(result.scene.scene_type, "classroom")
        self.assertTrue(result.content["公式定理"])


if __name__ == "__main__":
    unittest.main()
