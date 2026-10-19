"""The banner lands on top of the displayed page without covering content."""

import pymupdf
import pytest

from pdf2zh_next.watermark import stamp_banner

BANNER_URL = "https://example.com/landing"


def _banner(width=600, height=60, text="BANNER"):
    doc = pymupdf.open()
    page = doc.new_page(width=width, height=height)
    page.draw_rect(page.rect, color=None, fill=(0.9, 0.97, 0.92))
    if text:
        page.insert_text((12, 38), text, fontsize=22)
    page.insert_link(
        {"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(530, 5, 590, 55), "uri": BANNER_URL}
    )
    return doc.tobytes()


def _write(tmp_path, pages=1, width=612, height=792, rotation=0, cropbox=None, name="out.pdf"):
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page(width=width, height=height)
        page.insert_text((72, 100), "BODY", fontsize=30)
        if cropbox is not None:
            page.set_cropbox(cropbox)
        page.set_rotation(rotation)
    path = tmp_path / name
    doc.save(path)
    return path


def _word_box(page, word):
    for w in page.get_text("words"):
        if w[4] == word:
            return pymupdf.Rect(w[:4])
    raise AssertionError(f"{word!r} not on page")


def test_first_page_grows_by_banner_height_and_content_moves_below(tmp_path):
    path = _write(tmp_path, pages=2)

    assert stamp_banner(path, _banner(), columns=1, all_pages=False) == 1

    doc = pymupdf.open(path)
    strip = 60 * 612 / 600
    assert doc[0].rect.height == pytest.approx(792 + strip)
    assert doc[1].rect.height == pytest.approx(792)
    banner = _word_box(doc[0], "BANNER")
    body = _word_box(doc[0], "BODY")
    assert banner.y1 <= strip < body.y0
    assert [link["uri"] for link in doc[0].get_links()] == [BANNER_URL]
    assert doc[1].get_links() == []


def test_all_pages(tmp_path):
    path = _write(tmp_path, pages=3)

    assert stamp_banner(path, _banner(), columns=1, all_pages=True) == 3


def test_side_by_side_dual_gets_one_banner_per_column(tmp_path):
    path = _write(tmp_path, width=1224)

    stamp_banner(path, _banner(), columns=2, all_pages=False)

    page = pymupdf.open(path)[0]
    lefts = sorted(round(link["from"].x0) for link in page.get_links())
    # Link at x=530 of a 600-wide banner, scaled to a 612-wide column.
    assert lefts == [round(530 * 612 / 600), round(612 + 530 * 612 / 600)]


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_rotated_page_gets_banner_at_displayed_top(tmp_path, rotation):
    path = _write(tmp_path, rotation=rotation)
    before = pymupdf.open(path)[0].rect

    stamp_banner(path, _banner(), columns=1, all_pages=False)

    page = pymupdf.open(path)[0]
    strip = 60 * before.width / 600
    assert page.rect.width == pytest.approx(before.width)
    assert page.rect.height == pytest.approx(before.height + strip)
    # Rotation was folded into the content, so text boxes are display boxes.
    assert page.rotation == 0
    assert _word_box(page, "BANNER").y1 <= strip


def test_rotated_page_with_annotations_is_left_alone(tmp_path):
    path = _write(tmp_path, rotation=90)
    doc = pymupdf.open(path)
    doc[0].add_highlight_annot(pymupdf.Rect(72, 72, 160, 106))
    doc.saveIncr()
    doc.close()
    before = path.read_bytes()

    assert stamp_banner(path, _banner(), columns=1, all_pages=False) == 0
    assert path.read_bytes() == before


def test_content_hidden_by_crop_box_stays_hidden(tmp_path):
    path = _write(tmp_path, cropbox=pymupdf.Rect(40, 110, 572, 760))

    stamp_banner(path, _banner(text=None), columns=1, all_pages=False)

    page = pymupdf.open(path)[0]
    strip = 60 * 532 / 600
    pix = page.get_pixmap(clip=pymupdf.Rect(0, 0, page.rect.width, strip))
    # "BODY" sat above the crop box, inside the area the page grew into: only
    # the banner fill may show there, no dark glyph pixels.
    darkest = min(pix.samples[i] for i in range(0, len(pix.samples), pix.n))
    assert darkest > 200


def test_output_reopens_after_incremental_save(tmp_path):
    path = _write(tmp_path, pages=2)

    stamp_banner(path, _banner(), columns=1, all_pages=False)

    doc = pymupdf.open(path)
    assert doc.page_count == 2
    assert not doc.is_repaired


def _request(banner_b64, **kwargs):
    from pdf2zh_next.http_api import StreamTranslationRequest

    return StreamTranslationRequest(
        source_url="https://example.com/src.pdf",
        lang_in="en",
        lang_out="zh",
        watermark_banner_pdf=banner_b64,
        **kwargs,
    )


def test_stream_results_get_one_banner_per_column(tmp_path):
    import base64

    from pdf2zh_next.http_api import _stamp_results

    mono = _write(tmp_path, name="mono.pdf")
    dual = _write(tmp_path, width=1224, name="dual.pdf")

    request = _request(base64.b64encode(_banner()).decode())
    assert _stamp_results(request, str(mono), str(dual)) is True

    assert len(pymupdf.open(mono)[0].get_links()) == 1
    assert len(pymupdf.open(dual)[0].get_links()) == 2


def test_invalid_banner_leaves_results_untouched(tmp_path):
    from pdf2zh_next.http_api import _stamp_results

    mono = _write(tmp_path)
    before = mono.read_bytes()

    assert _stamp_results(_request("not base64!"), str(mono), None) is False
    assert mono.read_bytes() == before
