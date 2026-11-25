# -*- coding: utf-8 -*-
"""链家租房公开列表页爬虫。

说明：仅采集无需登录即可访问的城市租房首页公开信息，控制访问频率，
数据仅用于课程学习与统计分析。第二页起站点要求登录，程序会自动检测并停止翻页。
"""
from __future__ import annotations

import gzip
import random
import re
import time
from datetime import datetime
from http.cookiejar import CookieJar
from urllib import request
from urllib.parse import urljoin

import pandas as pd
from lxml import html as lxml_html

from config import (
    BASE_URL,
    CITIES,
    CITY_NAME,
    CRAWL_DELAY,
    HEADERS,
    RAW_DIR,
    REQUEST_TIMEOUT,
    RETRY_TIMES,
)

COLUMNS = [
    "house_code", "city", "region", "business_area", "community", "title",
    "rent_type", "rent_category", "rooms", "halls", "baths", "area",
    "orientation", "floor_level", "total_floors", "price", "tags",
    "maintain_text", "maintain_days", "brand", "url", "crawl_time", "data_source",
]


class LianjiaSession:
    """带 Cookie 的轻量会话，用于获取城市站点首页。"""

    def __init__(self, city_code: str):
        self.city_code = city_code
        self.base = f"https://{city_code}.lianjia.com"
        self.opener = request.build_opener(request.HTTPCookieProcessor(CookieJar()))

    def get(self, url: str, referer: str | None = None) -> str:
        headers = dict(HEADERS)
        headers["Referer"] = referer or f"{self.base}/zufang/"
        req = request.Request(url, headers=headers)
        last_err = None
        for attempt in range(RETRY_TIMES):
            try:
                with self.opener.open(req, timeout=REQUEST_TIMEOUT) as resp:
                    raw = resp.read()
                    if (resp.headers.get("Content-Encoding") or "").lower() == "gzip":
                        raw = gzip.decompress(raw)
                    return raw.decode("utf-8", errors="ignore")
            except Exception as exc:  # 网络抖动时重试
                last_err = exc
                if attempt < RETRY_TIMES - 1:
                    time.sleep(2 + attempt * 2)
        raise RuntimeError(f"请求失败 {url}: {last_err}")


def parse_maintain_days(text: str) -> float:
    """把“3天前维护 / 2小时前维护”等文本转换为天数。"""
    if not text:
        return float("nan")
    text = text.strip()
    if "刚刚" in text:
        return 0.0
    if "今天" in text:
        return 0.5
    if "昨天" in text:
        return 1.0
    if "前天" in text:
        return 2.0
    m = re.search(r"(\d+)\s*(分钟|小时|天|周|个月|月)", text)
    if not m:
        return float("nan")
    n, unit = int(m.group(1)), m.group(2)
    factor = {"分钟": 1 / 1440, "小时": 1 / 24, "天": 1, "周": 7, "个月": 30, "月": 30}
    return round(n * factor[unit], 2)


def parse_items(html_text: str, city_code: str, city_name: str) -> list[dict]:
    """解析一页房源列表，返回记录列表。"""
    if "ke-passport" in html_text and "content__list--item" not in html_text:
        return []  # 命中登录页
    doc = lxml_html.fromstring(html_text)
    nodes = doc.xpath('//div[@class="content__list--item"]')
    records = []
    crawl_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for node in nodes:
        try:
            title_node = node.xpath('.//p[contains(@class,"content__list--item--title")]/a')
            title = title_node[0].text_content().strip() if title_node else ""
            href = title_node[0].get("href", "") if title_node else ""
            url = urljoin(f"https://{city_code}.lianjia.com", href)
            house_code = node.get("data-house_code") or (url.rsplit("/", 1)[-1].replace(".html", ""))
            rent_category = node.get("data-bu_desc", "") or ""

            des_node = node.xpath('.//p[contains(@class,"content__list--item--des")]')
            des_text = ""
            region = business_area = community = ""
            if des_node:
                des_text = re.sub(r"\s+", " ", des_node[0].text_content()).strip()
                anchors = [a.text_content().strip() for a in des_node[0].xpath(".//a")]
                if anchors:
                    region = anchors[0]
                if len(anchors) >= 2:
                    business_area = anchors[1]
                if len(anchors) >= 3:
                    community = anchors[-1]

            segments = [s.strip() for s in des_text.split("/")]
            area = float("nan")
            orientation = ""
            for idx, seg in enumerate(segments):
                m_area = re.search(r"([\d.]+)\s*㎡", seg)
                if m_area:
                    area = float(m_area.group(1))
                    if idx + 1 < len(segments):
                        cand = segments[idx + 1].strip()
                        if re.fullmatch(r"[东南西北\s]{1,6}", cand):
                            orientation = re.sub(r"\s+", "", cand)
                    break
            if not orientation:
                m_ori = re.search(r"㎡\s*/\s*([东南西北\s]{1,6})\s*/", des_text)
                orientation = re.sub(r"\s+", "", m_ori.group(1)) if m_ori else ""

            m_layout = re.search(r"(\d+)室(\d+)厅(?:(\d+)卫)?", des_text)
            rooms = int(m_layout.group(1)) if m_layout else None
            halls = int(m_layout.group(2)) if m_layout else None
            baths = int(m_layout.group(3)) if m_layout and m_layout.group(3) else None

            m_floor = re.search(r"(低|中|高)楼层", des_text)
            floor_level = m_floor.group(1) if m_floor else ""
            m_total = re.search(r"[（(](\d+)层[）)]", des_text) or re.search(r"共(\d+)层", des_text)
            total_floors = int(m_total.group(1)) if m_total else None

            price_node = node.xpath('.//span[contains(@class,"content__list--item-price")]/em')
            price_text = price_node[0].text_content().strip() if price_node else ""
            m_price = re.search(r"(\d+)", price_text.replace(",", ""))
            price = float(m_price.group(1)) if m_price else float("nan")

            tags = [i.text_content().strip() for i in
                    node.xpath('.//p[contains(@class,"content__list--item--bottom")]/i')]
            tags = [t for t in tags if t]
            maintain_node = node.xpath('.//span[contains(@class,"content__list--item--time")]')
            maintain_text = maintain_node[0].text_content().strip() if maintain_node else ""
            brand_node = node.xpath('.//span[contains(@class,"brand")]')
            brand = brand_node[0].text_content().strip() if brand_node else ""

            rent_type = "合租" if ("合租" in title or "合租" in rent_category) else "整租"
            if not community and "独栋·" in title:
                community = title.split("独栋·", 1)[1].split(" ")[0].strip()
            if not rent_category:
                rent_category = "品牌公寓" if "独栋公寓" in "|".join(tags) else "普通租赁"

            records.append({
                "house_code": house_code,
                "city": city_name,
                "region": region,
                "business_area": business_area,
                "community": community,
                "title": title,
                "rent_type": rent_type,
                "rent_category": rent_category,
                "rooms": rooms,
                "halls": halls,
                "baths": baths,
                "area": area,
                "orientation": orientation,
                "floor_level": floor_level,
                "total_floors": total_floors,
                "price": price,
                "tags": "|".join(tags),
                "maintain_text": maintain_text,
                "maintain_days": parse_maintain_days(maintain_text),
                "brand": brand,
                "url": url,
                "crawl_time": crawl_time,
                "data_source": "链家租房公开列表页",
            })
        except Exception:
            continue
    return records


def crawl_city(city_code: str, city_name: str, pages: int = 1) -> list[dict]:
    session = LianjiaSession(city_code)
    base_url = BASE_URL.format(code=city_code)
    records: list[dict] = []
    for page in range(1, pages + 1):
        url = base_url if page == 1 else f"{base_url}pg{page}/"
        try:
            html_text = session.get(url, referer=base_url)
        except RuntimeError:
            break
        page_records = parse_items(html_text, city_code, city_name)
        if not page_records:
            break  # 登录墙或页面无数据，停止翻页
        records.extend(page_records)
        if page < pages:
            time.sleep(random.uniform(*CRAWL_DELAY))
    return records


def crawl_all(pages: int = 1, city_limit: int | None = None) -> pd.DataFrame:
    cities = CITIES[:city_limit] if city_limit else CITIES
    all_records: list[dict] = []
    for i, (code, name) in enumerate(cities, 1):
        try:
            rows = crawl_city(code, name, pages=pages)
        except Exception as exc:
            print(f"[{i}/{len(cities)}] {name} 抓取失败: {exc}")
            rows = []
        all_records.extend(rows)
        print(f"[{i}/{len(cities)}] {name}: 获取 {len(rows)} 条")
        if i < len(cities):
            time.sleep(random.uniform(*CRAWL_DELAY))

    df = pd.DataFrame(all_records, columns=COLUMNS)
    df = df.drop_duplicates(subset=["house_code"]).reset_index(drop=True)
    if not df.empty:
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        out = RAW_DIR / f"rent_raw_{stamp}.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        df.to_csv(RAW_DIR / "rent_raw_latest.csv", index=False, encoding="utf-8-sig")
        print(f"原始数据已保存: {out}（{len(df)} 条）")
    return df


def main() -> None:
    df = crawl_all(pages=1)
    print(df.groupby("city").size() if not df.empty else "未获取到数据")


if __name__ == "__main__":
    main()
