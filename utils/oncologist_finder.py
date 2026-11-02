"""Live, worldwide OpenStreetMap hospital discovery for OncoVision."""

from functools import lru_cache
import math
import os

import requests
from flask import Blueprint, jsonify, request

oncologist_finder = Blueprint("oncologist_finder", __name__)
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URLS = (
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
)
HEADERS = {
    "User-Agent": os.getenv("NOMINATIM_USER_AGENT", "OncoVision/1.0 (contact: support@oncovision.com)"),
    "Accept": "application/json",
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two coordinates in kilometres."""
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    value = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))


@lru_cache(maxsize=128)
def geocode_city(city_name):
    """Resolve any city worldwide; never force the search into Karnataka."""
    cleaned = " ".join(str(city_name or "").split())
    if not cleaned:
        return None
    response = requests.get(
        NOMINATIM_URL,
        params={"q": cleaned, "format": "jsonv2", "limit": 1, "addressdetails": 1},
        headers={**HEADERS, "Accept-Language": "en"}, timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    if not data:
        return None
    return float(data[0]["lat"]), float(data[0]["lon"])


def _element_coordinates(element):
    if element.get("lat") is not None and element.get("lon") is not None:
        return float(element["lat"]), float(element["lon"])
    center = element.get("center") or {}
    if center.get("lat") is not None and center.get("lon") is not None:
        return float(center["lat"]), float(center["lon"])
    return None


def _address(tags):
    if tags.get("addr:full"):
        return tags["addr:full"]
    return ", ".join(filter(None, (tags.get("addr:housenumber"), tags.get("addr:street"), tags.get("addr:suburb"))))


def _fetch_nominatim_hospitals(lat, lng, radius_km, limit):
    """Run a bounded live hospital search against OpenStreetMap Nominatim."""
    lat_delta = radius_km / 111.0
    lng_delta = radius_km / max(111.0 * math.cos(math.radians(lat)), 1.0)
    response = requests.get(
        NOMINATIM_URL,
        params={
            "q": "hospital",
            "format": "jsonv2",
            "limit": min(max(int(limit) * 2, 25), 50),
            "bounded": 1,
            "viewbox": f"{lng-lng_delta},{lat+lat_delta},{lng+lng_delta},{lat-lat_delta}",
            "addressdetails": 1,
            "extratags": 1,
        },
        headers={**HEADERS, "Accept-Language": "en"},
        timeout=(5, 15),
    )
    response.raise_for_status()
    results, seen = [], set()
    for place in response.json():
        name = (place.get("name") or str(place.get("display_name", "")).split(",", 1)[0]).strip()
        if not name:
            continue
        item_lat, item_lng = float(place["lat"]), float(place["lon"])
        distance = haversine_km(lat, lng, item_lat, item_lng)
        if distance > radius_km or name.casefold() in seen:
            continue
        seen.add(name.casefold())
        address, extra = place.get("address") or {}, place.get("extratags") or {}
        searchable = f"{name} {place.get('display_name', '')}".casefold()
        oncology = any(word in searchable for word in ("oncology", "cancer", "breast"))
        city = address.get("city") or address.get("town") or address.get("village") or address.get("county") or ""
        results.append({
            "name": name,
            "doctor": "Oncology Care Team" if oncology else "Hospital",
            "specialization": "Oncology Centre" if oncology else "Hospital",
            "address": place.get("display_name", ""),
            "city": city,
            "district": address.get("state_district") or address.get("county") or city,
            "lat": item_lat,
            "lng": item_lng,
            "distance_km": round(distance, 1),
            "rating": 0,
            "phone": extra.get("contact:phone") or extra.get("phone") or "",
            "website": extra.get("contact:website") or extra.get("website") or "",
            "source": "OpenStreetMap",
        })
    results.sort(key=lambda item: item["distance_km"])
    return results[: int(limit)]


@lru_cache(maxsize=128)
def fetch_real_hospitals(lat, lng, radius_km=25, limit=25):
    """Fetch named hospitals and oncology-specific clinics from live OSM data."""
    lat, lng, radius_km = float(lat), float(lng), float(radius_km)
    # Start with a fast bounded lookup. Overpass below provides a broader
    # fallback in sparse areas without making every button click wait for it.
    nominatim_available, quick_results = False, []
    try:
        quick_results = _fetch_nominatim_hospitals(lat, lng, radius_km, limit)
        nominatim_available = True
        if quick_results:
            return quick_results
    except (requests.RequestException, ValueError, KeyError):
        pass
    radius_m = int(radius_km * 1000)
    query = f"""
    [out:json][timeout:20];
    (
      nwr["amenity"="hospital"](around:{radius_m},{lat},{lng});
      nwr["healthcare"="hospital"](around:{radius_m},{lat},{lng});
      nwr["healthcare"="oncology"](around:{radius_m},{lat},{lng});
      nwr["healthcare:speciality"~"oncology|cancer|breast",i](around:{radius_m},{lat},{lng});
      nwr["amenity"="clinic"]["name"~"oncology|cancer|breast",i](around:{radius_m},{lat},{lng});
    );
    out center tags;
    """
    last_error, elements = None, None
    for endpoint in OVERPASS_URLS:
        try:
            response = requests.post(endpoint, data={"data": query}, headers=HEADERS, timeout=(4, 8))
            response.raise_for_status()
            elements = response.json().get("elements", [])
            break
        except (requests.RequestException, ValueError) as error:
            last_error = error
    if elements is None:
        if nominatim_available:
            return quick_results
        raise requests.RequestException(str(last_error or "OpenStreetMap hospital service is unavailable")) from last_error

    results, seen = [], set()
    for element in elements:
        tags = element.get("tags") or {}
        name = (tags.get("name:en") or tags.get("name") or "").strip()
        coordinates = _element_coordinates(element)
        if not name or not coordinates:
            continue
        item_lat, item_lng = coordinates
        identity = (name.casefold(), round(item_lat, 4), round(item_lng, 4))
        if identity in seen:
            continue
        seen.add(identity)
        searchable = " ".join((name, tags.get("healthcare:speciality", ""))).casefold()
        oncology = tags.get("healthcare") == "oncology" or any(word in searchable for word in ("oncology", "cancer", "breast"))
        city = tags.get("addr:city") or tags.get("addr:town") or tags.get("addr:village") or ""
        district = tags.get("addr:district") or tags.get("addr:county") or city
        results.append({
            "name": name,
            "doctor": "Oncology Care Team" if oncology else "Hospital",
            "specialization": "Oncology Centre" if oncology else "Hospital",
            "address": _address(tags), "city": city, "district": district,
            "lat": item_lat, "lng": item_lng,
            "distance_km": round(haversine_km(lat, lng, item_lat, item_lng), 1),
            "rating": 0,
            "phone": tags.get("contact:phone") or tags.get("phone") or "",
            "website": tags.get("contact:website") or tags.get("website") or "",
            "source": "OpenStreetMap",
        })
    results.sort(key=lambda item: item["distance_km"])
    return results[: int(limit)]


def _validated_radius(default=25):
    try:
        radius = float(request.args.get("radius_km", default))
    except (TypeError, ValueError) as error:
        raise ValueError("radius_km must be a number") from error
    if not 1 <= radius <= 100:
        raise ValueError("radius_km must be between 1 and 100")
    return radius


@oncologist_finder.get("/api/nearby-hospitals")
def nearby_hospitals():
    try:
        lat, lng = float(request.args.get("lat")), float(request.args.get("lng"))
        radius = _validated_radius()
        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            raise ValueError("Invalid coordinates")
    except (TypeError, ValueError) as error:
        return jsonify({"status": "error", "error": str(error) or "lat and lng are required"}), 400
    try:
        hospitals = fetch_real_hospitals(lat, lng, radius)
    except requests.RequestException:
        return jsonify({"status": "error", "message": "Live hospital data is temporarily unavailable. Please try again."}), 502
    response = jsonify({"status": "ok", "data_source": "OpenStreetMap live data", "user_location": {"lat": lat, "lng": lng}, "count": len(hospitals), "hospitals": hospitals})
    response.headers["Cache-Control"] = "no-store"
    return response


@oncologist_finder.get("/api/hospitals-by-city")
def hospitals_by_city():
    city = " ".join(request.args.get("city", "").split())
    if not city:
        return jsonify({"status": "error", "error": "city is required"}), 400
    try:
        radius, coordinates = _validated_radius(), geocode_city(city)
    except requests.RequestException:
        return jsonify({"status": "error", "message": "The location service is temporarily unavailable. Please try again."}), 502
    except ValueError as error:
        return jsonify({"status": "error", "error": str(error)}), 400
    if coordinates is None:
        return jsonify({"status": "not_found", "message": f"'{city}' was not found."}), 404
    lat, lng = coordinates
    try:
        hospitals = fetch_real_hospitals(lat, lng, radius)
    except requests.RequestException:
        return jsonify({"status": "error", "message": "Live hospital data is temporarily unavailable. Please try again."}), 502
    response = jsonify({"status": "ok", "data_source": "OpenStreetMap live data", "searched_city": city, "city_location": {"lat": lat, "lng": lng}, "count": len(hospitals), "hospitals": hospitals})
    response.headers["Cache-Control"] = "no-store"
    return response
