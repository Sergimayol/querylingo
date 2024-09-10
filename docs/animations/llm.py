"""
For generation video:
   - High quality:  manim -pqh llm.py GeneratingText
   - Low quality:  manim -pql llm.py GeneratingText
"""

from manim import Scene, Mobject, MoveToTarget, Tex, MobjectMatrix, Text, VGroup, ReplacementTransform, SurroundingRectangle, Title, Arrow
from manim import WHITE, BLACK, GREEN_E, LEFT, BLUE_E, MED_LARGE_BUFF, DOWN, MED_SMALL_BUFF, UP, IN, SMALL_BUFF


class BaseScene(Scene):

    ANIM_SPEED = 1.5

    def __init__(self):
        super().__init__()
        self.camera.background_color = WHITE
        Mobject.set_default(color=BLACK)

    def wait(self, num_secs=1):
        super().wait(num_secs * BaseScene.ANIM_SPEED)

    def play(self, *args, **kwargs):
        if "run_time" in kwargs:
            kwargs["run_time"] *= BaseScene.ANIM_SPEED
        else:
            kwargs["run_time"] = 1.2
        super().play(*args, **kwargs)

    def _make_matrix(self, embeddings, actual_h=None, actual_w=None):
        h, w = len(embeddings), len(embeddings[0])
        actual_h = actual_h or h
        actual_w = actual_w or w
        tr_input = []
        for row in embeddings:
            tr_row = [Tex(f"{e:.2f}") for e in row]
            if actual_w > w:
                tr_row = tr_row[:2] + [tr_row[-1]]
                tr_row.insert(-1, Tex("\\dots"))
            tr_input.append(tr_row)
        if actual_h > h:
            tr_input.append([Tex("\\vdots") for i in range(len(tr_input[-1]))])
        return MobjectMatrix(tr_input)

    def _make_row_labels(self, matrix, labels, color=GREEN_E, dir=LEFT):
        rows = matrix.get_rows()
        text_objs = []
        for row, label in zip(rows, labels):
            text_objs.append(Text(label, font_size=24).next_to(row, dir * 3).set_color(color))
        return VGroup(*text_objs)


class BaseSelfAttn(BaseScene):

    def __init__(self):
        super().__init__()
        self.sentence = "La traducción entre lenguaje"
        self.sentence_tokens = self.sentence.split(" ")
        self.n_embd = 768
        self.n_head = 12


class GeneratingText(BaseSelfAttn):

    def _make_tokens(self, tokens: list[str], color=BLUE_E) -> VGroup:
        grp = VGroup()
        for t in tokens:
            label = Tex(t)
            box = SurroundingRectangle(label, color=color)
            grp.add(VGroup(label, box))
        return grp

    def construct(self):
        title = Title("Generación de texto", match_underline_width_to_text=True)
        self.add(title)

        tf_label = Tex("Transformer")
        tf_box = SurroundingRectangle(tf_label, buff=MED_LARGE_BUFF).set_color(BLACK)
        tf_grp = VGroup(tf_label, tf_box).shift(DOWN)
        self.add(tf_grp)

        input_tokens = self._make_tokens(self.sentence_tokens)
        input_tokens.arrange(buff=MED_SMALL_BUFF)
        input_tokens.next_to(tf_grp, UP * 3)
        input_window = SurroundingRectangle(input_tokens).set_color(BLACK)

        max_window_sz = 5
        input_window_label = (
            Tex(f"Tamaño máx. de entrada = {max_window_sz}", font_size=28).next_to(input_window, UP, buff=SMALL_BUFF).align_to(input_window, LEFT)
        )

        in_arrow = Arrow(buff=0, start=input_window.get_bottom(), end=tf_grp.get_top())
        out_arrow = Arrow(buff=0, start=tf_grp.get_bottom(), end=tf_grp.get_bottom() + DOWN * 0.7)
        self.add(in_arrow, out_arrow, input_tokens, input_window, input_window_label)

        self.wait(1)
        q = "natural y sql con IA!"
        for i in q.split(" "):
            input_copy = input_tokens.copy()[max(0, len(input_tokens) - max_window_sz) :]
            for b in input_copy:
                b.generate_target()
                b.target.move_to(tf_grp.get_center())
                b.target.shift(IN)
                b.target.width = 0
                b.target.height = 0

            self.play(*[MoveToTarget(b) for b in input_copy])

            out_token = self._make_tokens([i], color=GREEN_E)[0]
            out_token.generate_target()
            out_token.move_to(tf_grp.get_center())
            out_token.width = 0
            out_token.height = 0
            out_token.target.next_to(tf_grp, DOWN * 4)

            self.play(ReplacementTransform(out_token, out_token.target))
            self.wait()
            out_token = out_token.target
            out_token.generate_target()
            for t in input_tokens:
                t.generate_target()
            new_input_tokens = VGroup(*[t.target for t in input_tokens], out_token.target).arrange(buff=MED_SMALL_BUFF).move_to(input_tokens)
            num_new_tokens = len(new_input_tokens)
            new_input_window = SurroundingRectangle(VGroup(*new_input_tokens[max(0, num_new_tokens - max_window_sz) :])).set_color(BLACK)
            self.play(
                *[ReplacementTransform(t, t.target) for t in input_tokens],
                ReplacementTransform(out_token, out_token.target),
                ReplacementTransform(input_window, new_input_window),
                input_window_label.animate.next_to(new_input_window, UP, buff=SMALL_BUFF).align_to(new_input_window, LEFT),
            )
            input_tokens = new_input_tokens
            input_window = new_input_window
            self.wait()

        self.wait(3)
