package format

import (
	"fmt"
	"html"
	"regexp"
	"strings"
)

func FormatHTML(text string) string {
    // Блочные кодовые фрагменты: Telegram HTML не принимает атрибуты у <code>,
    // поэтому удаляем class и экранируем содержимое.
    reCodeBlock := regexp.MustCompile("(?s)```(\\w+)?\n(.*?)\n```")
    text = reCodeBlock.ReplaceAllStringFunc(text, func(match string) string {
        parts := reCodeBlock.FindStringSubmatch(match)
        code := parts[2]
        return fmt.Sprintf(`<pre><code>%s</code></pre>`, html.EscapeString(strings.TrimSpace(code)))
    })

    // Инлайн-код: обязательно экранируем содержимое внутри тегов <code>.
    reInlineCode := regexp.MustCompile("`([^`]+)`")
    text = reInlineCode.ReplaceAllStringFunc(text, func(m string) string {
        // Срезаем обрамляющие обратные кавычки
        inner := m[1 : len(m)-1]
        return "<code>" + html.EscapeString(inner) + "</code>"
    })

    // Жирный текст **...**
    reBold := regexp.MustCompile(`\*\*(.+?)\*\*`)
    text = reBold.ReplaceAllString(text, `<b>$1</b>`)

    // Списки: заменяем маркер "* " на символ точки, до обработки курсивов,
    // чтобы звёздочки списка не интерпретировались как курсив.
    reListItem := regexp.MustCompile(`(?m)^(\*|-) `)
    text = reListItem.ReplaceAllString(text, "• ")

    // Курсив *...*: избегаем пересечения с **...** и не захватываем переносы строк.
    // Шаблон берёт предшествующий символ (не '*') для устойчивости.
    reItalic := regexp.MustCompile(`(^|[^*])\*([^*\n]+)\*`)
    text = reItalic.ReplaceAllString(text, `$1<i>$2</i>`)

    return text
}

func SplitMessage(message string, maxLen int) []string {
	if len(message) <= maxLen {
		return []string{message}
	}
	var parts []string
	for len(message) > 0 {
		if len(message) <= maxLen {
			parts = append(parts, message)
			break
		}
		splitPos := strings.LastIndex(message[:maxLen], "\n")
		if splitPos == -1 {
			splitPos = strings.LastIndex(message[:maxLen], " ")
		}
		if splitPos == -1 {
			splitPos = maxLen
		}
		parts = append(parts, message[:splitPos])
		message = strings.TrimSpace(message[splitPos:])
	}
	return parts
}


