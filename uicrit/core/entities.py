import ast

# Comment class to store each individual comment's details
class Comment:
    def __init__(self, text, bounding_box):
        self.text = text
        self.bounding_box = bounding_box

    def __repr__(self):
        return f"Comment(text={self.text[:30]}..., bounding_box={self.bounding_box})"

    def to_json(self):
        return {
            "comment": self.text,
            "bounding_box": self.bounding_box
        }

    @classmethod
    def from_json(cls, data):
        return cls(text=data["comment"], bounding_box=data.get("bounding_box"))


# Task class to store a task and its associated comments
class Task:
    def __init__(self, task_name, comments):
        self.task_name = task_name
        self.comments = comments

    def __repr__(self):
        return f"Task(task_name={self.task_name}, comments={self.comments})"

    def to_json(self):
        return {
            "task": self.task_name,
            "comments": [comment.to_json() for comment in self.comments]
        }

    @classmethod
    def from_json(cls, data):
        comments = [Comment.from_json(c) for c in data["comments"]]
        return cls(task_name=data["task"], comments=comments)


# UICritData class to store rico_id and a list of tasks
class UICritData:
    def __init__(self, rico_id):
        self.rico_id = rico_id
        self.tasks = []

    def add_task(self, task_name, comments_source, comments_str):
        """Parses and adds a task with its comments."""
        comments_list = ast.literal_eval(comments_str)
        comments_source_list = ast.literal_eval(comments_source)
        parsed_comments = []

        # Ensure the number of comments matches the number of sources
        for idx, comment in enumerate(comments_list):
            # Extract bounding box
            if 'Bounding Box:' in comment:
                parts = comment.split("Bounding Box:")
                text = parts[0].strip().split('\n', 1)[1]             
                bbox = [float(x) for x in parts[1].strip(' []').split(',')]
            else:
                text = comment.strip()
                bbox = None

            parsed_comments.append(Comment(text, bbox))

        # Add the task to the tasks list
        task = Task(task_name, parsed_comments)
        self.tasks.append(task)

    def __repr__(self):
        return f"UICritData(rico_id={self.rico_id}, tasks={self.tasks})"

    def to_json(self):
        return {
            "id": self.rico_id,
            "tasks": [task.to_json() for task in self.tasks]
        }

    @classmethod
    def from_json(cls, data):
        obj = cls(rico_id=data["id"])
        obj.tasks = [Task.from_json(t) for t in data["tasks"]]
        return obj


