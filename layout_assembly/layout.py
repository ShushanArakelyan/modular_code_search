from layout_assembly.utils import TestActionModuleWrapper


class LayoutNode:
    def __init__(self):
        self.node_type = None
        self.node_value = None
        self.children = []
        self.parent = None


class LayoutNet:
    def __init__(self, scoring_module, action_module):
        self.scoring_module = scoring_module
        self.action_module = action_module

    def forward(self, ccg_parse, code):
        tree = self.construct_layout(ccg_parse)
        tree = self.remove_concats(tree)
        _, output = self.process_node(tree, code)
        return output

    def process_node(self, node, code, parent_module=None):
        if node.node_type == 'action':
            action_module = ActionModuleWrapper(self.action_module)
            action_module.param = node.node_value
            for child in node.children:
                action_module, _ = self.process_node(child, code, action_module)
            output = action_module.forward(code)
            if parent_module:
                parent_module.add_input(output)
            return parent_module, output
        elif node.node_type == 'scoring':
            output = self.scoring_module.forward(node.node_value, code)
            parent_module.add_input(output)
            return parent_module, output
        elif node.node_type == 'preposition':
            parent_module.add_preposition(node.node_value)
            for child in node.children:
                self.process_node(child, code, parent_module)
            return parent_module, None

    @staticmethod
    def remove_concats(tree):
        stack = [tree]
        while stack:
            node = stack.pop()
            new_children = []
            new_node = None
            for child in node.children:
                if child.node_type == 'scoring':
                    if new_node is None:
                        new_node = LayoutNode()
                        new_node.node_type = 'scoring'
                        new_node.parent = node.parent
                        new_node.node_value = child.node_value
                    else:
                        new_node.node_value += ' ' + child.node_value
                else:
                    if new_node:
                        new_children.append(new_node)
                        new_node = None
                    new_children.append(child)
            if new_node:
                new_children.append(new_node)
            node.children = new_children
            stack.extend(new_children)
        return tree

    @staticmethod
    def construct_layout(ccg_parse):
        parts = ccg_parse.split(' ')[::-1]
        tree = LayoutNode()
        node = tree
        stack = []
        parent = None
        while parts:
            current_part = parts.pop()
            if len(current_part) == 0:
                continue
            if current_part == '@Action':
                node.node_type = 'action'
                parts.pop()  # opening bracket
                node.node_value = parts.pop()
                node.parent = parent
                stack.append(node)
                if parent:
                    parent.children.append(node)
                parent = node
                node = LayoutNode()
            elif current_part == '@Concat' or current_part == '@Num':
                node.node_type = 'scoring'
                parts.pop()  # opening bracket
                node.node_value = parts.pop()
                node.parent = parent
                stack.append(node)
                parent.children.append(node)

                node = LayoutNode()
            elif current_part.startswith('@'):
                node.node_type = 'preposition'
                node.node_value = current_part[1:].lower()
                parts.pop()  # opening bracket
                node.parent = parent
                stack.append(node)
                parent.children.append(node)

                parent = node
                node = LayoutNode()
            elif current_part.startswith(')'):
                current_part = list(current_part)
                while current_part:
                    stack.pop()
                    current_part.pop()
                if stack:
                    parent = stack[-1]
            else:
                node.node_type = 'scoring'
                node.node_value = current_part

                node.parent = parent
                parent.children.append(node)

                node = LayoutNode()
        return tree