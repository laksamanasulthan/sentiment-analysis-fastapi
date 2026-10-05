import re

CATEGORIES = ["Customer Support", "Product Quality", "Delivery & Shipping", "Pricing & Billing", "General"]

CATEGORY_PATTERNS = [
    (
        "Customer Support",
        re.compile(r"customer|support|helpdesk|staff|agent|\bcs\b|respon|layanan|pelanggan|ramah|complaint|keluhan|pengaduan", re.IGNORECASE),
    ),
    (
        "Delivery & Shipping",
        re.compile(r"deliver|shipping|shipment|courier|package|parcel|ongkir|pengiriman|paket|kurir|ekspedisi|terlambat", re.IGNORECASE),
    ),
    (
        "Pricing & Billing",
        re.compile(r"price|pricing|cost|billing|payment|charge|refund|expensive|cheap|discount|harga|mahal|murah|bayar|tagihan|promo|diskon|tarif", re.IGNORECASE),
    ),
    (
        "Product Quality",
        re.compile(r"product|quality|item|battery|screen|durable|broken|defect|produk|kualitas|barang|baterai|layar|rusak|cacat", re.IGNORECASE),
    ),
]


def detect_category(text: str) -> str:
    for category, pattern in CATEGORY_PATTERNS:
        if pattern.search(text):
            return category
    return "General"
