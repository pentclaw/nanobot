from nanobot.channels.feishu import _extract_interactive_content


def test_extract_interactive_content_handles_flat_elements_list() -> None:
    content = {
        "elements": [
            {"tag": "markdown", "content": "card body"},
            {
                "tag": "button",
                "text": {"content": "open"},
                "url": "https://example.com",
            },
        ]
    }

    assert _extract_interactive_content(content) == [
        "card body",
        "open",
        "link: https://example.com",
    ]
