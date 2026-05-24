"""ANSI-aware Markdown renderer for terminal output."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode

from harnify_tui.terminal_image import getCapabilities, hyperlink, isImageLine
from harnify_tui.tui import Component
from harnify_tui.utils import applyBackgroundToLine, visibleWidth, wrapTextWithAnsi

type StyleFn = Callable[[str], str]
type HighlightCodeFn = Callable[[str, str | None], list[str]]

_BARE_URL_RE = r"https?://[^\s<>()]+[^\s<>().,!?:;\"')\]]"
_EMAIL_RE = r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])"
_AUTOLINK_RE = re.compile(f"(?P<url>{_BARE_URL_RE})|(?P<email>{_EMAIL_RE})")

_MARKDOWN_PARSER = MarkdownIt("commonmark").enable("table").enable("strikethrough")


@dataclass(slots=True)
class DefaultTextStyle:
    color: StyleFn | None = None
    bgColor: StyleFn | None = None
    bold: bool = False
    italic: bool = False
    strikethrough: bool = False
    underline: bool = False


@dataclass(slots=True)
class MarkdownTheme:
    heading: StyleFn
    link: StyleFn
    linkUrl: StyleFn
    code: StyleFn
    codeBlock: StyleFn
    codeBlockBorder: StyleFn
    quote: StyleFn
    quoteBorder: StyleFn
    hr: StyleFn
    listBullet: StyleFn
    bold: StyleFn
    italic: StyleFn
    strikethrough: StyleFn
    underline: StyleFn
    highlightCode: HighlightCodeFn | None = None
    codeBlockIndent: str | None = None


@dataclass(slots=True)
class InlineStyleContext:
    applyText: StyleFn
    stylePrefix: str


class Markdown(Component):
    def __init__(
        self,
        text: str,
        paddingX: int,
        paddingY: int,
        theme: MarkdownTheme,
        defaultTextStyle: DefaultTextStyle | None = None,
    ) -> None:
        self.text = text
        self.paddingX = paddingX
        self.paddingY = paddingY
        self.theme = theme
        self.defaultTextStyle = defaultTextStyle
        self.defaultStylePrefix: str | None = None
        self.cachedText: str | None = None
        self.cachedWidth: int | None = None
        self.cachedLines: list[str] | None = None
        self._sourceLines: list[str] = []

    def setText(self, text: str) -> None:
        self.text = text
        self.invalidate()

    def invalidate(self) -> None:
        self.cachedText = None
        self.cachedWidth = None
        self.cachedLines = None
        self.defaultStylePrefix = None

    def render(self, width: int) -> list[str]:
        if self.cachedLines is not None and self.cachedText == self.text and self.cachedWidth == width:
            return self.cachedLines

        content_width = max(1, width - self.paddingX * 2)
        if not self.text or self.text.strip() == "":
            result: list[str] = []
            self.cachedText = self.text
            self.cachedWidth = width
            self.cachedLines = result
            return result

        normalized_text = self.text.replace("\t", "   ")
        self._sourceLines = normalized_text.split("\n")
        root = SyntaxTreeNode(_MARKDOWN_PARSER.parse(normalized_text))

        rendered_lines: list[str] = []
        for index, node in enumerate(root.children or []):
            next_type = root.children[index + 1].type if index + 1 < len(root.children) else None
            rendered_lines.extend(self.renderBlock(node, content_width, next_type))

        wrapped_lines: list[str] = []
        for line in rendered_lines:
            if isImageLine(line):
                wrapped_lines.append(line)
            else:
                wrapped_lines.extend(wrapTextWithAnsi(line, content_width))

        left_margin = " " * self.paddingX
        right_margin = " " * self.paddingX
        bg_fn = self.defaultTextStyle.bgColor if self.defaultTextStyle else None
        content_lines: list[str] = []
        for line in wrapped_lines:
            if isImageLine(line):
                content_lines.append(line)
                continue

            line_with_margins = left_margin + line + right_margin
            if bg_fn is not None:
                content_lines.append(applyBackgroundToLine(line_with_margins, width, bg_fn))
            else:
                visible_len = visibleWidth(line_with_margins)
                content_lines.append(line_with_margins + (" " * max(0, width - visible_len)))

        empty_line = " " * width
        empty_lines: list[str] = []
        for _ in range(self.paddingY):
            empty_lines.append(applyBackgroundToLine(empty_line, width, bg_fn) if bg_fn is not None else empty_line)

        result = [*empty_lines, *content_lines, *empty_lines]
        self.cachedText = self.text
        self.cachedWidth = width
        self.cachedLines = result
        return result if result else [""]

    def applyDefaultStyle(self, text: str) -> str:
        if self.defaultTextStyle is None:
            return text

        styled = text
        if self.defaultTextStyle.color is not None:
            styled = self.defaultTextStyle.color(styled)
        if self.defaultTextStyle.bold:
            styled = self.theme.bold(styled)
        if self.defaultTextStyle.italic:
            styled = self.theme.italic(styled)
        if self.defaultTextStyle.strikethrough:
            styled = self.theme.strikethrough(styled)
        if self.defaultTextStyle.underline:
            styled = self.theme.underline(styled)
        return styled

    def getDefaultStylePrefix(self) -> str:
        if self.defaultTextStyle is None:
            return ""
        if self.defaultStylePrefix is not None:
            return self.defaultStylePrefix

        sentinel = "\x00"
        styled = sentinel
        if self.defaultTextStyle.color is not None:
            styled = self.defaultTextStyle.color(styled)
        if self.defaultTextStyle.bold:
            styled = self.theme.bold(styled)
        if self.defaultTextStyle.italic:
            styled = self.theme.italic(styled)
        if self.defaultTextStyle.strikethrough:
            styled = self.theme.strikethrough(styled)
        if self.defaultTextStyle.underline:
            styled = self.theme.underline(styled)

        sentinel_index = styled.find(sentinel)
        self.defaultStylePrefix = styled[:sentinel_index] if sentinel_index >= 0 else ""
        return self.defaultStylePrefix

    def getStylePrefix(self, styleFn: StyleFn) -> str:
        sentinel = "\x00"
        styled = styleFn(sentinel)
        sentinel_index = styled.find(sentinel)
        return styled[:sentinel_index] if sentinel_index >= 0 else ""

    def getDefaultInlineStyleContext(self) -> InlineStyleContext:
        return InlineStyleContext(applyText=self.applyDefaultStyle, stylePrefix=self.getDefaultStylePrefix())

    def renderBlock(
        self,
        node: SyntaxTreeNode,
        width: int,
        nextType: str | None = None,
        styleContext: InlineStyleContext | None = None,
    ) -> list[str]:
        lines: list[str] = []

        match node.type:
            case "heading":
                heading_level = int(node.tag[1:]) if node.tag.startswith("h") else 1
                heading_prefix = f"{'#' * heading_level} "

                if heading_level == 1:
                    def heading_style_fn(text: str) -> str:
                        return self.theme.heading(self.theme.bold(self.theme.underline(text)))
                else:
                    def heading_style_fn(text: str) -> str:
                        return self.theme.heading(self.theme.bold(text))

                heading_style_context = InlineStyleContext(
                    applyText=heading_style_fn,
                    stylePrefix=self.getStylePrefix(heading_style_fn),
                )
                heading_text = self.renderInlineNodes(node.children or [], heading_style_context)
                styled_heading = (
                    f"{heading_style_fn(heading_prefix)}{heading_text}" if heading_level >= 3 else heading_text
                )
                lines.append(styled_heading)
                if nextType is not None:
                    lines.append("")

            case "paragraph":
                lines.append(self.renderInlineNodes(node.children or [], styleContext))
                if nextType not in {None, "bullet_list", "ordered_list"}:
                    lines.append("")

            case "text":
                lines.append(self.renderInlineNodes([node], styleContext))

            case "fence" | "code_block":
                indent = self.theme.codeBlockIndent or "  "
                lines.append(self.theme.codeBlockBorder(f"```{node.info or ''}"))
                if self.theme.highlightCode is not None:
                    highlighted_lines = self.theme.highlightCode(node.content, node.info or None)
                    for highlighted in highlighted_lines:
                        lines.append(f"{indent}{highlighted}")
                else:
                    code_lines = node.content.rstrip("\n").split("\n") if node.content else [""]
                    for code_line in code_lines:
                        lines.append(f"{indent}{self.theme.codeBlock(code_line)}")
                lines.append(self.theme.codeBlockBorder("```"))
                if nextType is not None:
                    lines.append("")

            case "bullet_list" | "ordered_list":
                lines.extend(self.renderList(node, 0, width, styleContext))

            case "table":
                lines.extend(self.renderTable(node, width, nextType, styleContext))

            case "blockquote":
                def quote_style(text: str) -> str:
                    return self.theme.quote(self.theme.italic(text))

                quote_style_prefix = self.getStylePrefix(quote_style)
                quote_content_width = max(1, width - 2)
                quote_style_context = InlineStyleContext(
                    applyText=lambda text: text,
                    stylePrefix=quote_style_prefix,
                )

                rendered_quote_lines: list[str] = []
                for index, child in enumerate(node.children or []):
                    next_child_type = node.children[index + 1].type if index + 1 < len(node.children) else None
                    rendered_quote_lines.extend(
                        self.renderBlock(child, quote_content_width, next_child_type, quote_style_context)
                    )

                while rendered_quote_lines and rendered_quote_lines[-1] == "":
                    rendered_quote_lines.pop()

                for quote_line in rendered_quote_lines:
                    styled_line = self.applyQuoteStyle(quote_line, quote_style, quote_style_prefix)
                    for wrapped_line in wrapTextWithAnsi(styled_line, quote_content_width):
                        lines.append(self.theme.quoteBorder("│ ") + wrapped_line)

                if nextType is not None:
                    lines.append("")

            case "hr":
                lines.append(self.theme.hr("─" * min(width, 80)))
                if nextType is not None:
                    lines.append("")

            case "html_block":
                if node.content.strip():
                    lines.append(self.applyDefaultStyle(node.content.strip()))

            case _:
                if node.content:
                    lines.append(node.content)

        return lines

    def applyQuoteStyle(self, line: str, quoteStyle: StyleFn, quoteStylePrefix: str) -> str:
        if not quoteStylePrefix:
            return quoteStyle(line)
        line_with_reapplied_style = line.replace("\x1b[0m", f"\x1b[0m{quoteStylePrefix}")
        return quoteStyle(line_with_reapplied_style)

    def renderInlineNodes(
        self,
        nodes: list[SyntaxTreeNode],
        styleContext: InlineStyleContext | None = None,
    ) -> str:
        result = ""
        resolved = styleContext or self.getDefaultInlineStyleContext()
        apply_text = resolved.applyText
        style_prefix = resolved.stylePrefix

        def apply_text_with_newlines(text: str) -> str:
            return "\n".join(apply_text(segment) for segment in text.split("\n"))

        for node in nodes:
            match node.type:
                case "text":
                    result += self.renderAutolinkText(node.content, resolved)

                case "inline":
                    result += self.renderInlineNodes(node.children or [], resolved)

                case "paragraph":
                    result += self.renderInlineNodes(node.children or [], resolved)

                case "strong":
                    result += self.theme.bold(self.renderInlineNodes(node.children or [], resolved)) + style_prefix

                case "em":
                    result += self.theme.italic(self.renderInlineNodes(node.children or [], resolved)) + style_prefix

                case "code_inline":
                    result += self.theme.code(node.content) + style_prefix

                case "link":
                    link_text = self.renderInlineNodes(node.children or [], resolved)
                    link_text_plain = self.inlinePlainText(node.children or [])
                    href = node.attrs.get("href", "")
                    result += self.renderLink(link_text, link_text_plain, href, style_prefix)

                case "s":
                    result += (
                        self.theme.strikethrough(self.renderInlineNodes(node.children or [], resolved)) + style_prefix
                    )

                case "html_inline":
                    result += apply_text_with_newlines(node.content)

                case "softbreak" | "hardbreak":
                    result += "\n"

                case _:
                    if node.children:
                        result += self.renderInlineNodes(node.children, resolved)
                    elif node.content:
                        result += apply_text_with_newlines(node.content)

        while style_prefix and result.endswith(style_prefix):
            result = result[: -len(style_prefix)]
        return result

    def renderAutolinkText(self, text: str, styleContext: InlineStyleContext) -> str:
        if "\n" in text:
            return "\n".join(self.renderAutolinkText(part, styleContext) for part in text.split("\n"))

        result = ""
        last_end = 0
        for match in _AUTOLINK_RE.finditer(text):
            start, end = match.span()
            result += styleContext.applyText(text[last_end:start])
            url = match.group("url")
            email = match.group("email")
            value = url or email or ""
            href = value if url is not None else f"mailto:{value}"
            styled_link = self.theme.link(self.theme.underline(value))
            result += self.renderLink(styled_link, value, href, styleContext.stylePrefix, alreadyStyled=True)
            last_end = end
        result += styleContext.applyText(text[last_end:])
        return result

    def renderLink(
        self,
        linkText: str,
        linkTextPlain: str,
        href: str,
        stylePrefix: str,
        *,
        alreadyStyled: bool = False,
    ) -> str:
        styled_link = linkText if alreadyStyled else self.theme.link(self.theme.underline(linkText))
        if getCapabilities().hyperlinks:
            return hyperlink(styled_link, href) + stylePrefix

        href_for_comparison = href[7:] if href.startswith("mailto:") else href
        if linkTextPlain == href or linkTextPlain == href_for_comparison:
            return styled_link + stylePrefix
        return styled_link + self.theme.linkUrl(f" ({href})") + stylePrefix

    def inlinePlainText(self, nodes: list[SyntaxTreeNode]) -> str:
        parts: list[str] = []
        for node in nodes:
            match node.type:
                case "text" | "code_inline" | "html_inline":
                    parts.append(node.content)
                case "softbreak" | "hardbreak":
                    parts.append("\n")
                case _:
                    if node.children:
                        parts.append(self.inlinePlainText(node.children))
                    elif node.content:
                        parts.append(node.content)
        return "".join(parts)

    def renderList(
        self,
        node: SyntaxTreeNode,
        depth: int,
        width: int,
        styleContext: InlineStyleContext | None = None,
    ) -> list[str]:
        lines: list[str] = []
        indent = "    " * depth
        start_number = int(node.attrs.get("start", 1)) if node.type == "ordered_list" else 1

        for index, item in enumerate(node.children or []):
            bullet = f"{start_number + index}. " if node.type == "ordered_list" else "- "
            first_prefix = indent + self.theme.listBullet(bullet)
            continuation_prefix = indent + (" " * visibleWidth(bullet))
            item_width = max(1, width - visibleWidth(first_prefix))
            rendered_any_line = False

            for child in item.children or []:
                if child.type in {"bullet_list", "ordered_list"}:
                    lines.extend(self.renderList(child, depth + 1, width, styleContext))
                    rendered_any_line = True
                    continue

                child_lines = self.renderBlock(child, item_width, None, styleContext)
                for line in child_lines:
                    for wrapped_line in wrapTextWithAnsi(line, item_width):
                        line_prefix = continuation_prefix if rendered_any_line else first_prefix
                        lines.append(line_prefix + wrapped_line)
                        rendered_any_line = True

            if not rendered_any_line:
                lines.append(first_prefix)

        return lines

    def getLongestWordWidth(self, text: str, maxWidth: int | None = None) -> int:
        longest = 0
        for word in (segment for segment in text.split() if segment):
            longest = max(longest, visibleWidth(word))
        return min(longest, maxWidth) if maxWidth is not None else longest

    def wrapCellText(self, text: str, maxWidth: int) -> list[str]:
        return wrapTextWithAnsi(text, max(1, maxWidth))

    def renderTable(
        self,
        node: SyntaxTreeNode,
        availableWidth: int,
        nextType: str | None = None,
        styleContext: InlineStyleContext | None = None,
    ) -> list[str]:
        lines: list[str] = []
        if not node.children:
            return lines

        header_rows = node.children[0].children if node.children and node.children[0].type == "thead" else []
        body_section = node.children[1] if len(node.children) > 1 and node.children[1].type == "tbody" else None
        body_rows = body_section.children if body_section is not None else []
        if not header_rows:
            return lines

        header_row = header_rows[0]
        num_cols = len(header_row.children or [])
        if num_cols == 0:
            return lines

        border_overhead = 3 * num_cols + 1
        available_for_cells = availableWidth - border_overhead
        if available_for_cells < num_cols:
            fallback_lines = wrapTextWithAnsi(self.rawSourceForNode(node), availableWidth)
            if nextType is not None:
                fallback_lines.append("")
            return fallback_lines

        max_unbroken_word_width = 30
        natural_widths = [0] * num_cols
        min_word_widths = [1] * num_cols

        for index, cell in enumerate(header_row.children or []):
            text = self.renderInlineNodes(cell.children or [], styleContext)
            natural_widths[index] = visibleWidth(text)
            min_word_widths[index] = max(1, self.getLongestWordWidth(text, max_unbroken_word_width))

        for row in body_rows:
            for index, cell in enumerate(row.children or []):
                text = self.renderInlineNodes(cell.children or [], styleContext)
                natural_widths[index] = max(natural_widths[index], visibleWidth(text))
                min_word_widths[index] = max(
                    min_word_widths[index],
                    self.getLongestWordWidth(text, max_unbroken_word_width),
                )

        min_column_widths = list(min_word_widths)
        min_cells_width = sum(min_column_widths)
        if min_cells_width > available_for_cells:
            min_column_widths = [1] * num_cols
            remaining = available_for_cells - num_cols
            if remaining > 0:
                total_weight = sum(max(0, width - 1) for width in min_word_widths)
                growth = [
                    int((max(0, width - 1) / total_weight) * remaining) if total_weight > 0 else 0
                    for width in min_word_widths
                ]
                for index, width_value in enumerate(growth):
                    min_column_widths[index] += width_value

                allocated = sum(growth)
                leftover = remaining - allocated
                for index in range(num_cols):
                    if leftover <= 0:
                        break
                    min_column_widths[index] += 1
                    leftover -= 1

            min_cells_width = sum(min_column_widths)

        total_natural_width = sum(natural_widths) + border_overhead
        if total_natural_width <= availableWidth:
            column_widths = [
                max(width_value, min_column_widths[index]) for index, width_value in enumerate(natural_widths)
            ]
        else:
            total_grow_potential = sum(
                max(0, width_value - min_column_widths[index]) for index, width_value in enumerate(natural_widths)
            )
            extra_width = max(0, available_for_cells - min_cells_width)
            column_widths = []
            for index, min_width in enumerate(min_column_widths):
                natural_width = natural_widths[index]
                width_delta = max(0, natural_width - min_width)
                grow = int((width_delta / total_grow_potential) * extra_width) if total_grow_potential > 0 else 0
                column_widths.append(min_width + grow)

            allocated = sum(column_widths)
            remaining = available_for_cells - allocated
            while remaining > 0:
                grew = False
                for index in range(num_cols):
                    if remaining <= 0:
                        break
                    if column_widths[index] < natural_widths[index]:
                        column_widths[index] += 1
                        remaining -= 1
                        grew = True
                if not grew:
                    break

        lines.append(f"┌─{'─┬─'.join('─' * width_value for width_value in column_widths)}─┐")

        header_cell_lines = [
            self.wrapCellText(self.renderInlineNodes(cell.children or [], styleContext), column_widths[index])
            for index, cell in enumerate(header_row.children or [])
        ]
        header_line_count = max((len(cell_lines) for cell_lines in header_cell_lines), default=0)
        for line_index in range(header_line_count):
            row_parts: list[str] = []
            for column_index, cell_lines in enumerate(header_cell_lines):
                text = cell_lines[line_index] if line_index < len(cell_lines) else ""
                padded = text + (" " * max(0, column_widths[column_index] - visibleWidth(text)))
                row_parts.append(self.theme.bold(padded))
            lines.append(f"│ {' │ '.join(row_parts)} │")

        separator_line = f"├─{'─┼─'.join('─' * width_value for width_value in column_widths)}─┤"
        lines.append(separator_line)

        for row_index, row in enumerate(body_rows):
            row_cell_lines = [
                self.wrapCellText(self.renderInlineNodes(cell.children or [], styleContext), column_widths[index])
                for index, cell in enumerate(row.children or [])
            ]
            row_line_count = max((len(cell_lines) for cell_lines in row_cell_lines), default=0)
            for line_index in range(row_line_count):
                row_parts = []
                for column_index, cell_lines in enumerate(row_cell_lines):
                    text = cell_lines[line_index] if line_index < len(cell_lines) else ""
                    row_parts.append(text + (" " * max(0, column_widths[column_index] - visibleWidth(text))))
                lines.append(f"│ {' │ '.join(row_parts)} │")
            if row_index < len(body_rows) - 1:
                lines.append(separator_line)

        lines.append(f"└─{'─┴─'.join('─' * width_value for width_value in column_widths)}─┘")
        if nextType is not None:
            lines.append("")
        return lines

    def rawSourceForNode(self, node: SyntaxTreeNode) -> str:
        if node.map is None:
            return node.content
        start, end = node.map
        return "\n".join(self._sourceLines[start:end])


__all__ = ["DefaultTextStyle", "Markdown", "MarkdownTheme"]
