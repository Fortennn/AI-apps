"""Програмні правила: перевірка змісту вилучених даних."""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"


@dataclass
class Issue:
    """Проблема, знайдена правилом."""
    field: str
    rule: str
    message: str
    severity: str = "error"


def load_reference() -> tuple[dict, list[dict]]:
    """Прочитати реквізити своєї компанії й довідник постачальників."""
    company = json.loads((REFERENCE_DIR / "company.json").read_text(encoding="utf-8"))
    suppliers = json.loads((REFERENCE_DIR / "suppliers.json").read_text(encoding="utf-8"))
    return company, suppliers


def verify_iban_checksum(iban: str) -> bool:
    """Перевірка контрольної суми IBAN за стандартом ISO 13616 (MOD 97-10)."""
    cleaned = iban.replace(" ", "").upper().strip()
    if len(cleaned) != 29 or not cleaned.startswith("UA"):
        return False
    # Переносимо перші 4 символи в кінець
    rearranged = cleaned[4:] + cleaned[:4]
    # Замінюємо літери на цифри: A=10, ..., Z=35
    digits = []
    for ch in rearranged:
        if ch.isdigit():
            digits.append(ch)
        elif "A" <= ch <= "Z":
            digits.append(str(ord(ch) - ord("A") + 10))
        else:
            return False
    num_str = "".join(digits)
    try:
        return int(num_str) % 97 == 1
    except ValueError:
        return False


def verify_edrpou_checksum(code: str) -> bool:
    """Перевірка контрольного розряду коду ЄДРПОУ (8 цифр)."""
    if len(code) != 8 or not code.isdigit():
        return False
    d = [int(x) for x in code]
    val = int(code)
    if val < 30000000 or val > 60000000:
        w1, w2 = [1, 2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8, 9]
    else:
        w1, w2 = [7, 1, 2, 3, 4, 5, 6], [9, 3, 4, 5, 6, 7, 8]
    s1 = sum(d[i] * w1[i] for i in range(7))
    c = s1 % 11
    if c == 10:
        s2 = sum(d[i] * w2[i] for i in range(7))
        c = s2 % 11
        if c == 10:
            c = 0
    return d[7] == c


def verify_rnokpp_checksum(code: str) -> bool:
    """Перевірка контрольного розряду РНОКПП (10 цифр для ФОП)."""
    if len(code) != 10 or not code.isdigit():
        return False
    d = [int(x) for x in code]
    w = [-1, 5, 7, 9, 4, 6, 10, 5, 7]
    s = sum(d[i] * w[i] for i in range(9))
    c = (s % 11) % 10
    return d[9] == c


def check(document: dict) -> list[Issue]:
    """Застосувати програмні правила до вилученого документа."""
    issues: list[Issue] = []
    company, suppliers = load_reference()

    # 1. Тип документа
    doc_type = str(document.get("document_type") or "").strip().lower()
    if doc_type != "рахунок":
        issues.append(Issue(
            field="document_type",
            rule="document_type",
            message=f"Документ не є рахунком на оплату (виявлено: {document.get('document_type')})",
            severity="error"
        ))
        # Якщо це не рахунок, повертаємо одразу помилку типу документа
        return issues

    # 2. Обовʼязкові поля (missing_required)
    req_top_fields = [
        ("number", "Номер документа"),
        ("date", "Дата документа"),
        ("total_without_vat", "Разом без ПДВ"),
        ("vat", "Сума ПДВ"),
        ("total", "Усього до сплати"),
    ]
    tot_no_vat_val = document.get("total_without_vat")
    tot_val = document.get("total")
    is_non_vat = (tot_no_vat_val is not None and tot_val is not None and tot_no_vat_val == tot_val)

    for key, name in req_top_fields:
        if key == "vat" and is_non_vat:
            continue
        if document.get(key) is None or str(document.get(key)).strip() == "":
            issues.append(Issue(
                field=key,
                rule="missing_required",
                message=f"Відсутнє обовʼязкове поле: {name}",
                severity="error"
            ))

    supp = document.get("supplier") or {}
    for skey, sname in [("name", "Назва постачальника"), ("code", "Код постачальника"), ("iban", "IBAN постачальника")]:
        if not supp.get(skey):
            issues.append(Issue(
                field=f"supplier.{skey}",
                rule="missing_required",
                message=f"Відсутнє обовʼязкове поле постачальника: {sname}",
                severity="error"
            ))

    buyer = document.get("buyer") or {}
    for bkey, bname in [("name", "Назва покупця"), ("code", "Код покупця")]:
        if not buyer.get(bkey):
            issues.append(Issue(
                field=f"buyer.{bkey}",
                rule="missing_required",
                message=f"Відсутнє обовʼязкове поле покупця: {bname}",
                severity="error"
            ))

    items = document.get("items") or []
    if not items:
        issues.append(Issue(
            field="items",
            rule="missing_required",
            message="Відсутні товарні позиції у рахунку",
            severity="error"
        ))

    # 3. Перевірка покупця (покупець — саме наша компанія «Сузірʼя Рітейл»)
    buyer_code = str(buyer.get("code") or "").strip()
    buyer_name = str(buyer.get("name") or "").strip().lower()
    our_code = str(company.get("code") or "").strip()
    if buyer_code and buyer_code != our_code:
        issues.append(Issue(
            field="buyer.code",
            rule="buyer",
            message=f"Рахунок виставлено іншій компанії (код {buyer_code}, наш код {our_code})",
            severity="error"
        ))
    elif "сузір" not in buyer_name:
        issues.append(Issue(
            field="buyer.name",
            rule="buyer",
            message="Рахунок виставлено не ТОВ «Сузірʼя Рітейл»",
            severity="error"
        ))

    # 4. Перевірка кодів ЄДРПОУ / РНОКПП (контрольні розряди)
    supp_code = str(supp.get("code") or "").strip()
    if supp_code:
        if len(supp_code) == 8:
            if not verify_edrpou_checksum(supp_code):
                issues.append(Issue(
                    field="supplier.code",
                    rule="code_checksum",
                    message="Контрольний розряд коду ЄДРПОУ постачальника не зійшовся",
                    severity="error"
                ))
        elif len(supp_code) == 10:
            if not verify_rnokpp_checksum(supp_code):
                issues.append(Issue(
                    field="supplier.code",
                    rule="code_checksum",
                    message="Контрольний розряд РНОКПП постачальника не зійшовся",
                    severity="error"
                ))
        else:
            issues.append(Issue(
                field="supplier.code",
                rule="code_checksum",
                message=f"Невірна довжина коду постачальника ({len(supp_code)} цифр замість 8 чи 10)",
                severity="error"
            ))

    if buyer_code and len(buyer_code) == 8 and not verify_edrpou_checksum(buyer_code):
        issues.append(Issue(
            field="buyer.code",
            rule="code_checksum",
            message="Контрольний розряд ЄДРПОУ покупця не зійшовся",
            severity="error"
        ))

    # 5. IBAN постачальника: формат і контрольна сума ISO 13616
    supp_iban = str(supp.get("iban") or "").replace(" ", "").strip()
    if supp_iban:
        if not verify_iban_checksum(supp_iban):
            issues.append(Issue(
                field="supplier.iban",
                rule="iban_checksum",
                message="Невірна контрольна сума або формат IBAN постачальника",
                severity="error"
            ))

    # 6. Звірка постачальника та IBAN із довідником (reference/suppliers.json)
    matched_supplier = None
    if supp_code:
        matched_supplier = next((s for s in suppliers if s.get("code") == supp_code), None)

    if matched_supplier is None:
        # Спробуємо знайти за назвою
        supp_n = str(supp.get("name") or "").lower()
        matched_supplier = next((s for s in suppliers if s.get("name", "").lower() in supp_n or supp_n in s.get("name", "").lower()), None)

    if matched_supplier is None:
        issues.append(Issue(
            field="supplier.code",
            rule="iban_registry",
            message="Постачальник відсутній у довіднику перевірених постачальників",
            severity="warning"
        ))
    else:
        reg_iban = str(matched_supplier.get("iban") or "").replace(" ", "").strip()
        if supp_iban and supp_iban != reg_iban:
            issues.append(Issue(
                field="supplier.iban",
                rule="iban_registry",
                message=f"IBAN постачальника не збігається із зареєстрованим у довіднику (в довіднику: {reg_iban})",
                severity="error"
            ))

    # 7. Арифметика рядків (quantity * price == amount)
    for idx, it in enumerate(items):
        q = it.get("quantity")
        p = it.get("price")
        a = it.get("amount")
        if q is not None and p is not None and a is not None:
            try:
                dec_q = Decimal(str(q))
                dec_p = Decimal(str(p))
                dec_a = Decimal(str(a))
                expected_a = (dec_q * dec_p).quantize(Decimal("0.01"))
                if abs(expected_a - dec_a) > Decimal("0.02"):
                    issues.append(Issue(
                        field=f"items.{idx}.amount",
                        rule="arithmetic_line",
                        message=f"Сума рядка {idx + 1} ({a}) не дорівнює добутку кількості {q} та ціни {p} (= {expected_a})",
                        severity="error"
                    ))
            except (InvalidOperation, TypeError):
                issues.append(Issue(
                    field=f"items.{idx}.amount",
                    rule="arithmetic_line",
                    message=f"Некоректне числове значення в рядку {idx + 1}",
                    severity="error"
                ))

    # 8. Арифметика підсумків та ПДВ
    tot_no_vat = document.get("total_without_vat")
    vat_val = document.get("vat")
    tot_val = document.get("total")

    if items and tot_no_vat is not None:
        try:
            dec_tot_no_vat = Decimal(str(tot_no_vat))
            sum_items = sum(Decimal(str(it["amount"])) for it in items if it.get("amount") is not None)
            if abs(sum_items - dec_tot_no_vat) > Decimal("0.02"):
                issues.append(Issue(
                    field="total_without_vat",
                    rule="arithmetic_total",
                    message=f"Підсумок без ПДВ ({tot_no_vat}) не дорівнює сумі рядків ({sum_items:.2f})",
                    severity="error"
                ))
        except (InvalidOperation, TypeError):
            pass

    if tot_no_vat is not None and vat_val is not None:
        try:
            dec_tot_no_vat = Decimal(str(tot_no_vat))
            dec_vat = Decimal(str(vat_val))
            # Визначаємо, чи є постачальник платником ПДВ
            is_vat_payer = True
            if matched_supplier and matched_supplier.get("vat_payer") is False:
                is_vat_payer = False
            elif dec_vat == Decimal("0.00"):
                is_vat_payer = False

            if not is_vat_payer:
                if dec_vat != Decimal("0.00"):
                    issues.append(Issue(
                        field="vat",
                        rule="arithmetic_vat",
                        message=f"Постачальник не є платником ПДВ, але вказано ПДВ: {dec_vat}",
                        severity="error"
                    ))
            else:
                expected_vat = (dec_tot_no_vat * Decimal("0.20")).quantize(Decimal("0.01"))
                if abs(expected_vat - dec_vat) > Decimal("0.02"):
                    issues.append(Issue(
                        field="vat",
                        rule="arithmetic_vat",
                        message=f"Сума ПДВ ({dec_vat}) не відповідає 20% ({expected_vat})",
                        severity="error"
                    ))
        except (InvalidOperation, TypeError):
            pass

    if tot_no_vat is not None and vat_val is not None and tot_val is not None:
        try:
            dec_tot_no_vat = Decimal(str(tot_no_vat))
            dec_vat = Decimal(str(vat_val))
            dec_tot = Decimal(str(tot_val))
            expected_tot = dec_tot_no_vat + dec_vat
            if abs(expected_tot - dec_tot) > Decimal("0.02"):
                issues.append(Issue(
                    field="total",
                    rule="arithmetic_total",
                    message=f"Разом до сплати ({dec_tot}) не дорівнює сумі без ПДВ та ПДВ ({expected_tot:.2f})",
                    severity="error"
                ))
        except (InvalidOperation, TypeError):
            pass

    # 9. Перевірка дат
    doc_date = document.get("date")
    valid_until = document.get("valid_until")
    parsed_date = None

    if doc_date:
        try:
            parsed_date = datetime.strptime(str(doc_date), "%Y-%m-%d").date()
        except ValueError:
            issues.append(Issue(
                field="date",
                rule="dates",
                message=f"Некоректний формат дати документа: {doc_date}",
                severity="error"
            ))

    if valid_until:
        try:
            parsed_valid = datetime.strptime(str(valid_until), "%Y-%m-%d").date()
            if parsed_date and parsed_valid < parsed_date:
                issues.append(Issue(
                    field="valid_until",
                    rule="dates",
                    message=f"Строк дії рахунку ({valid_until}) передує даті виставлення ({doc_date})",
                    severity="error"
                ))
        except ValueError:
            issues.append(Issue(
                field="valid_until",
                rule="dates",
                message=f"Некоректний формат строку дії: {valid_until}",
                severity="error"
            ))

    return issues
