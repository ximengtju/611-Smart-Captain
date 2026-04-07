class TaskDispatcher:
    def __init__(self, nav_skill):
        #初始化 目前的skill 后续增加其他skill
        self.nav_skill = nav_skill

    def dispatch(self, task: dict):
        #先取出 任务类型
        task_type = task.get("task_type")

        if task_type == "navigate_to_point": #导航任务
            return self.nav_skill.execute(task["target"]) #调度器就会把 target 传给导航 skill 执行。

        if task_type == "reject":
            return {
                "status": "rejected",
                "reason": task.get("reason", "unknown")
            }

        return {
            "status": "failed",
            "reason": f"unsupported task_type: {task_type}"
        }