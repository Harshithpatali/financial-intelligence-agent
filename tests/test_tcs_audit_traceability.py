from evaluation.traceability import page_for_anchor, page_for_excerpt


def test_page_for_anchor_reads_plain_pdf_extraction_page_markers():
    text = "Page 1 of 3\nOverview\nPage 2 of 3\nRevenue from operations was 100 crore."
    assert page_for_anchor(text, "Revenue from operations") == 2


def test_page_for_anchor_reads_html_page_markers():
    text = "Intro\n<!-- PAGE: 7 -->\nTotal assets were 500 crore."
    assert page_for_anchor(text, "Total assets") == 7


def test_page_for_excerpt_resolves_page_from_source_excerpt():
    text = "Page 1 of 2\nOld content.\nPage 2 of 2\nThe company reported operating cash flow of 42 crore."
    excerpt = "The company reported operating cash flow of 42 crore."
    assert page_for_excerpt(text, excerpt) == 2


def test_page_traceability_is_unknown_when_no_page_marker_exists():
    assert page_for_anchor("Revenue from operations was 100 crore.", "Revenue") is None
