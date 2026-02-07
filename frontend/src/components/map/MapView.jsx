// src/components/map/MapView.jsx
import { useCallback } from "react";
import { MapContainer, TileLayer, useMapEvents, Marker, Popup } from "react-leaflet";
import L from "leaflet";

// Fix default marker icons in webpack/vite (otherwise broken icon path)
const defaultIcon = L.icon({
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});
L.Marker.prototype.options.icon = defaultIcon;

const USA_CENTER = [39.5, -98.35];
const USA_ZOOM = 4;

function MapClickHandler({ onSelectPoint }) {
  useMapEvents({
    click: (e) => {
      const { lat, lng } = e.latlng;
      onSelectPoint?.(lat, lng);
    },
  });
  return null;
}

export default function MapView({ selectedPoint, onSelectPoint }) {
  const position = selectedPoint
    ? [selectedPoint.lat, selectedPoint.lon]
    : USA_CENTER;

  return (
    <div style={{ position: "absolute", inset: 0, background: "#020503" }}>
      <MapContainer
        center={USA_CENTER}
        zoom={USA_ZOOM}
        style={{ height: "100%", width: "100%", position: "absolute", inset: 0 }}
        scrollWheelZoom={true}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <MapClickHandler onSelectPoint={onSelectPoint} />
        {selectedPoint && (
          <Marker position={[selectedPoint.lat, selectedPoint.lon]}>
            <Popup>
              {selectedPoint.lat.toFixed(5)}, {selectedPoint.lon.toFixed(5)}
            </Popup>
          </Marker>
        )}
      </MapContainer>
    </div>
  );
}
