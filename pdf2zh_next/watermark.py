"""Stamp a promotional banner above the content of a translated PDF.

The banner arrives as a one-page PDF that the caller designed and rendered, so
this module knows nothing about its copy, language or link target. It grows the
page at its visual top by the banner's height and draws the banner in the new
strip: the original content is never covered, whatever its margins look like.

Links on the banner page (a browser renders ``<a href>`` as link annotations)
are carried over, because ``show_pdf_page`` copies page content only.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf


def stamp_banner(
    pdf_path: Path,
    banner_pdf: bytes,
    *,
    columns: int,
    all_pages: bool,
) -> int:
    """Stamp ``banner_pdf`` onto ``pdf_path`` in place; return the pages stamped.

    ``columns`` is how many source pages sit side by side on one output page
    (2 for a side-by-side dual PDF); each column gets its own banner scaled to
    the column width.
    """
    banner_doc = pymupdf.open(stream=banner_pdf, filetype="pdf")
    banner_page = banner_doc[0]
    links = [
        link for link in banner_page.get_links() if link.get("kind") == pymupdf.LINK_URI
    ]

    doc = pymupdf.open(pdf_path)
    page_numbers = range(doc.page_count) if all_pages else range(min(1, doc.page_count))
    stamped = sum(
        _stamp_page(doc[pno], banner_doc, banner_page.rect, links, columns)
        for pno in page_numbers
    )
    if stamped == 0:
        doc.close()
        banner_doc.close()
        return 0

    # The outputs are large and only a page or two changed: append the changes
    # instead of rewriting the whole file whenever the document allows it.
    if doc.can_save_incrementally():
        doc.saveIncr()
        doc.close()
    else:
        tmp = pdf_path.with_name(pdf_path.name + ".stamped")
        doc.save(tmp, garbage=1, deflate=True)
        doc.close()
        tmp.replace(pdf_path)
    banner_doc.close()
    return stamped


def _stamp_page(
    page: pymupdf.Page,
    banner_doc: pymupdf.Document,
    banner_rect: pymupdf.Rect,
    links: list[dict],
    columns: int,
) -> bool:
    # Fold /Rotate into the content first (the page looks the same), so the
    # displayed top is the unrotated top. Growing a rotated page at its left or
    # bottom edge moves the coordinate origin, and PyMuPDF then misplaces
    # what is drawn afterwards.
    if page.rotation:
        # remove_rotation() puts annotations and links in the wrong place.
        # BabelDOC strips /Annots from its input, so its outputs carry none; if
        # that ever changes, leave the page alone rather than break it.
        if page.first_annot or page.first_link or page.first_widget:
            return False
        page.remove_rotation()

    column_width = page.rect.width / columns
    scale = column_width / banner_rect.width
    strip = banner_rect.height * scale
    _grow_top(page, strip)

    for column in range(columns):
        left = column * column_width
        target = pymupdf.Rect(left, 0, left + column_width, strip)
        # A crop box smaller than the old media box leaves hidden content in the
        # area the page just grew into; paint over it before the banner.
        page.draw_rect(target, color=None, fill=(1, 1, 1), overlay=True)
        page.show_pdf_page(target, banner_doc, 0)
        for link in links:
            area = link["from"]
            page.insert_link(
                {
                    "kind": pymupdf.LINK_URI,
                    "from": pymupdf.Rect(
                        left + area.x0 * scale,
                        area.y0 * scale,
                        left + area.x1 * scale,
                        area.y1 * scale,
                    ),
                    "uri": link["uri"],
                }
            )
    return True


def _grow_top(page: pymupdf.Page, strip: float) -> None:
    """Enlarge the page by ``strip`` points at its top edge.

    Starts from the visible area (the crop box) so a cropped page stays cropped,
    and works in PDF coordinates, where ``set_mediabox`` expects its rectangle.
    Setting the media box drops the crop box, so the result is exactly the old
    visible area plus the new strip.
    """
    visible = page.cropbox * ~page.transformation_matrix
    visible.normalize()
    visible.y1 += strip
    page.set_mediabox(visible)
