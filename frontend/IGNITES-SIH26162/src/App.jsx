import { useEffect, useMemo, useState } from "react";
import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Popup,
  useMap,
} from "react-leaflet";

import "leaflet/dist/leaflet.css";
import "./App.css";

/* =========================================================
   SAMPLE DATA
========================================================= */

const detections = [
  {
    id: "TH-001",
    classification: "Industrial Fire",
    latitude: 16.5062,
    longitude: 80.648,
    location: "Vijayawada Region, Andhra Pradesh",
    timestamp: "2026-09-15 16:25",
    brightness: 362.8,
    frp: 28.4,
    confidence: 96,
    satellite: "VIIRS",
    risk: "Critical",
    persistence: 4,
    nearestFeature: "Industrial Facility",
    distance: 0.7,
    landCover: "Built-up / Industrial",
    ndvi: 0.08,
    nbr: -0.31,
    dnbr: 0.48,
  },
  {
    id: "TH-002",
    classification: "Agricultural Fire",
    latitude: 17.385,
    longitude: 78.4867,
    location: "Hyderabad Region, Telangana",
    timestamp: "2026-09-15 14:42",
    brightness: 337.6,
    frp: 14.7,
    confidence: 91,
    satellite: "VIIRS",
    risk: "High",
    persistence: 2,
    nearestFeature: "Agricultural Land",
    distance: 0.3,
    landCover: "Cropland",
    ndvi: 0.21,
    nbr: -0.12,
    dnbr: 0.25,
  },
  {
    id: "TH-003",
    classification: "Wildfire",
    latitude: 20.2961,
    longitude: 85.8245,
    location: "Bhubaneswar Region, Odisha",
    timestamp: "2026-09-15 12:18",
    brightness: 348.9,
    frp: 19.2,
    confidence: 89,
    satellite: "MODIS",
    risk: "High",
    persistence: 3,
    nearestFeature: "Forest Area",
    distance: 1.2,
    landCover: "Forest",
    ndvi: 0.64,
    nbr: -0.28,
    dnbr: 0.51,
  },
  {
    id: "TH-004",
    classification: "Gas Flare",
    latitude: 23.0225,
    longitude: 72.5714,
    location: "Ahmedabad Region, Gujarat",
    timestamp: "2026-09-14 23:11",
    brightness: 381.2,
    frp: 32.6,
    confidence: 94,
    satellite: "VIIRS",
    risk: "Medium",
    persistence: 8,
    nearestFeature: "Oil & Gas Facility",
    distance: 0.2,
    landCover: "Industrial",
    ndvi: 0.02,
    nbr: -0.04,
    dnbr: 0.07,
  },
  {
    id: "TH-005",
    classification: "Mining Activity",
    latitude: 23.2599,
    longitude: 77.4126,
    location: "Bhopal Region, Madhya Pradesh",
    timestamp: "2026-09-14 18:37",
    brightness: 329.4,
    frp: 9.8,
    confidence: 84,
    satellite: "VIIRS",
    risk: "Medium",
    persistence: 5,
    nearestFeature: "Mining Area",
    distance: 0.9,
    landCover: "Mining / Barren",
    ndvi: 0.05,
    nbr: 0.11,
    dnbr: 0.03,
  },
  {
    id: "TH-006",
    classification: "Agricultural Fire",
    latitude: 15.8281,
    longitude: 78.0373,
    location: "Kurnool Region, Andhra Pradesh",
    timestamp: "2026-09-14 17:22",
    brightness: 341.7,
    frp: 16.2,
    confidence: 92,
    satellite: "VIIRS",
    risk: "High",
    persistence: 3,
    nearestFeature: "Agricultural Land",
    distance: 0.5,
    landCover: "Cropland",
    ndvi: 0.19,
    nbr: -0.18,
    dnbr: 0.31,
  },
  {
    id: "TH-007",
    classification: "Industrial Fire",
    latitude: 12.9716,
    longitude: 77.5946,
    location: "Bengaluru Region, Karnataka",
    timestamp: "2026-09-14 13:54",
    brightness: 355.1,
    frp: 23.1,
    confidence: 93,
    satellite: "VIIRS",
    risk: "High",
    persistence: 2,
    nearestFeature: "Industrial Facility",
    distance: 1.0,
    landCover: "Built-up",
    ndvi: 0.12,
    nbr: -0.22,
    dnbr: 0.34,
  },
  {
    id: "TH-008",
    classification: "Gas Flare",
    latitude: 19.076,
    longitude: 72.8777,
    location: "Mumbai Region, Maharashtra",
    timestamp: "2026-09-13 21:08",
    brightness: 372.4,
    frp: 26.4,
    confidence: 97,
    satellite: "VIIRS",
    risk: "Low",
    persistence: 10,
    nearestFeature: "Oil & Gas Facility",
    distance: 0.4,
    landCover: "Industrial",
    ndvi: 0.01,
    nbr: -0.02,
    dnbr: 0.04,
  },
  {
    id: "TH-009",
    classification: "Wildfire",
    latitude: 14.6819,
    longitude: 77.6006,
    location: "Anantapur Region, Andhra Pradesh",
    timestamp: "2026-09-13 16:16",
    brightness: 334.2,
    frp: 12.8,
    confidence: 87,
    satellite: "MODIS",
    risk: "Medium",
    persistence: 3,
    nearestFeature: "Forest Area",
    distance: 2.1,
    landCover: "Shrub / Forest",
    ndvi: 0.52,
    nbr: -0.16,
    dnbr: 0.29,
  },
  {
    id: "TH-010",
    classification: "Mining Activity",
    latitude: 18.5204,
    longitude: 73.8567,
    location: "Pune Region, Maharashtra",
    timestamp: "2026-09-12 19:44",
    brightness: 321.8,
    frp: 7.3,
    confidence: 82,
    satellite: "VIIRS",
    risk: "Low",
    persistence: 6,
    nearestFeature: "Mining Area",
    distance: 0.6,
    landCover: "Barren / Mining",
    ndvi: 0.04,
    nbr: 0.14,
    dnbr: 0.02,
  },
];
const persistenceHistory = {
  "TH-001": [
    { date: "Sep 08", frp: 24.1 },
    { date: "Sep 11", frp: 26.8 },
    { date: "Sep 13", frp: 27.5 },
    { date: "Sep 15", frp: 28.4 },
  ],
  "TH-002": [
    { date: "Sep 12", frp: 12.1 },
    { date: "Sep 15", frp: 14.7 },
  ],
  "TH-003": [
    { date: "Sep 10", frp: 15.4 },
    { date: "Sep 13", frp: 18.1 },
    { date: "Sep 15", frp: 19.2 },
  ],
  "TH-004": [
    { date: "Sep 07", frp: 30.2 },
    { date: "Sep 09", frp: 31.4 },
    { date: "Sep 11", frp: 33.1 },
    { date: "Sep 12", frp: 31.8 },
    { date: "Sep 13", frp: 32.2 },
    { date: "Sep 14", frp: 32.6 },
  ],
  "TH-005": [
    { date: "Sep 05", frp: 6.1 },
    { date: "Sep 08", frp: 7.2 },
    { date: "Sep 10", frp: 8.4 },
    { date: "Sep 12", frp: 9.1 },
    { date: "Sep 14", frp: 9.8 },
  ],
  "TH-006": [
    { date: "Sep 09", frp: 13.8 },
    { date: "Sep 11", frp: 15.1 },
    { date: "Sep 14", frp: 16.2 },
  ],
  "TH-007": [
    { date: "Sep 12", frp: 21.6 },
    { date: "Sep 14", frp: 23.1 },
  ],
  "TH-008": [
    { date: "Sep 04", frp: 22.4 },
    { date: "Sep 06", frp: 24.2 },
    { date: "Sep 08", frp: 25.1 },
    { date: "Sep 10", frp: 26.7 },
    { date: "Sep 11", frp: 25.8 },
    { date: "Sep 12", frp: 27.2 },
    { date: "Sep 13", frp: 26.1 },
    { date: "Sep 14", frp: 26.4 },
  ],
  "TH-009": [
    { date: "Sep 08", frp: 10.4 },
    { date: "Sep 11", frp: 11.9 },
    { date: "Sep 13", frp: 12.8 },
  ],
  "TH-010": [
    { date: "Sep 02", frp: 5.8 },
    { date: "Sep 05", frp: 6.2 },
    { date: "Sep 08", frp: 6.7 },
    { date: "Sep 10", frp: 7.0 },
    { date: "Sep 12", frp: 7.3 },
    { date: "Sep 14", frp: 7.3 },
  ],
};

const osmContext = {
  "TH-001": {
    nearestFeature: "Industrial Facility",
    locality: "Vijayawada industrial corridor",
    zoneType: "Industrial",
    insideIndustrialZone: true,
  },
  "TH-002": {
    nearestFeature: "Agricultural Land",
    locality: "Sultan Bazar region",
    zoneType: "Agricultural",
    insideIndustrialZone: false,
  },
  "TH-003": {
    nearestFeature: "Forest Area",
    locality: "Bhubaneswar forest fringe",
    zoneType: "Forest",
    insideIndustrialZone: false,
  },
  "TH-004": {
    nearestFeature: "Oil & Gas Facility",
    locality: "Ahmedabad industrial region",
    zoneType: "Industrial",
    insideIndustrialZone: true,
  },
  "TH-005": {
    nearestFeature: "Mining Area",
    locality: "Bhopal mining region",
    zoneType: "Mining",
    insideIndustrialZone: false,
  },
  "TH-006": {
    nearestFeature: "Agricultural Land",
    locality: "Kurnool agricultural belt",
    zoneType: "Agricultural",
    insideIndustrialZone: false,
  },
  "TH-007": {
    nearestFeature: "Industrial Facility",
    locality: "Bengaluru industrial region",
    zoneType: "Industrial",
    insideIndustrialZone: true,
  },
  "TH-008": {
    nearestFeature: "Oil & Gas Facility",
    locality: "Mumbai industrial region",
    zoneType: "Industrial",
    insideIndustrialZone: true,
  },
  "TH-009": {
    nearestFeature: "Forest Area",
    locality: "Anantapur forest fringe",
    zoneType: "Forest",
    insideIndustrialZone: false,
  },
  "TH-010": {
    nearestFeature: "Mining Area",
    locality: "Pune mining region",
    zoneType: "Mining",
    insideIndustrialZone: false,
  },
};

function getClassificationReason(detection) {
  const reasons = {
    "Industrial Fire":
      "Industrial land-use match + elevated FRP + built-up surroundings",
    "Agricultural Fire":
      "Agricultural land-use match + low-to-moderate FRP + short detection history",
    Wildfire:
      "Forest proximity + vegetation signature + repeated thermal detections",
    "Gas Flare":
      "Oil and gas facility match + high FRP + fixed-location persistence",
    "Mining Activity":
      "Mining-area match + barren land signature + recurring low-to-moderate FRP",
  };

  return (
    reasons[detection.classification] ||
    "Multi-source thermal and spatial analysis"
  );
}
const categories = [
  { name: "Industrial Fire", color: "#e07a3f" },
  { name: "Agricultural Fire", color: "#d8a63a" },
  { name: "Wildfire", color: "#d95858" },
  { name: "Gas Flare", color: "#4f8fc9" },
  { name: "Mining Activity", color: "#8b78c7" },
];

function getColor(name) {
  return categories.find((item) => item.name === name)?.color || "#78909c";
}

function MapFocus({ detection, zoom = 17 }) {
  const map = useMap();

  useEffect(() => {
    if (detection) {
      map.flyTo([detection.latitude, detection.longitude], zoom, {
        duration: 1.3,
      });
    }
  }, [detection, map, zoom]);

  return null;
}

function RiskBadge({ risk }) {
  return (
    <span className={`risk-badge risk-${risk.toLowerCase()}`}>{risk}</span>
  );
}

function StatCard({ title, value, subtitle, accent = false }) {
  return (
    <div className={`stat-card ${accent ? "stat-accent" : ""}`}>
      <span>{title}</span>
      <strong>{value}</strong>
      <small>{subtitle}</small>
    </div>
  );
}

function PageHeading({ eyebrow, title, description }) {
  return (
    <div className="page-heading">
      <span>{eyebrow}</span>
      <h1>{title}</h1>
      <p>{description}</p>
    </div>
  );
}

function DetailRow({ label, value }) {
  return (
    <div className="detail-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
function PersistenceTimeline({ detection }) {
  const history = persistenceHistory[detection.id] || [];
  const maxFRP = Math.max(...history.map((item) => item.frp), 1);

  return (
    <div className="details enhanced-details">
      <div className="section-heading-row">
        <div>
          <h4>PERSISTENCE TIMELINE</h4>
          <p className="section-caption">
            {detection.persistence} detection
            {detection.persistence === 1 ? "" : "s"} across the observation
            period
          </p>
        </div>

        <span
          className={`persistence-status ${
            detection.persistence >= 4 ? "recurring" : "limited"
          }`}
        >
          {detection.persistence >= 4 ? "Recurring" : "Limited"}
        </span>
      </div>

      <div className="persistence-chart">
        {history.map((item) => (
          <div
            className="persistence-point"
            key={`${detection.id}-${item.date}`}
          >
            <div className="persistence-bar-wrapper">
              <div
                className="persistence-bar"
                style={{
                  height: `${Math.max((item.frp / maxFRP) * 100, 18)}%`,
                }}
                title={`${item.date}: ${item.frp} MW FRP`}
              />
            </div>

            <span>{item.date}</span>
          </div>
        ))}
      </div>

      <p className="timeline-note">
        FRP by detection:{" "}
        {history.map((item) => `${item.date} (${item.frp} MW)`).join(" · ")}
      </p>
    </div>
  );
}

function ClassificationEvidence({ detection }) {
  return (
    <div className="classification-evidence">
      <div className="evidence-icon">✦</div>

      <div>
        <span>WHY THIS CLASSIFICATION?</span>
        <p>{getClassificationReason(detection)}</p>
      </div>
    </div>
  );
}

function OSMContext({ detection }) {
  const context = osmContext[detection.id];

  if (!context) return null;

  return (
    <div className="details osm-details">
      <div className="section-heading-row">
        <div>
          <h4>OPENSTREETMAP CONTEXT</h4>
          <p className="section-caption">Land-use and nearby feature context</p>
        </div>

        <span className="osm-badge">OSM</span>
      </div>

      <DetailRow
        label="Nearest tagged feature"
        value={context.nearestFeature}
      />
      <DetailRow label="Locality" value={context.locality} />
      <DetailRow label="Land-use zone" value={context.zoneType} />

      <DetailRow
        label="Inside industrial zone"
        value={context.insideIndustrialZone ? "Yes" : "No"}
      />
    </div>
  );
}

function SeverityLegend() {
  return (
    <div className="severity-legend">
      <strong>SEVERITY</strong>

      <span>
        <i className="severity-dot low" />
        Low — monitor
      </span>

      <span>
        <i className="severity-dot medium" />
        Medium — review
      </span>

      <span>
        <i className="severity-dot high" />
        High — priority
      </span>

      <span>
        <i className="severity-dot critical" />
        Critical — urgent
      </span>
    </div>
  );
}
/* =========================================================
   APP
========================================================= */

export default function App() {
  const [page, setPage] = useState("dashboard");

  const [selectedDetection, setSelectedDetection] = useState(null);

  const [classification, setClassification] = useState("All");

  /* User typed values */
  const [frpInput, setFrpInput] = useState("");
  const [persistenceInput, setPersistenceInput] = useState("");

  /* Applied exact values */
  const [appliedFRP, setAppliedFRP] = useState("");
  const [appliedPersistence, setAppliedPersistence] = useState("");

  const filteredDetections = useMemo(() => {
    return detections.filter((item) => {
      const classificationMatch =
        classification === "All" || item.classification === classification;

      const frpMatch = appliedFRP === "" || item.frp === Number(appliedFRP);

      const persistenceMatch =
        appliedPersistence === "" ||
        item.persistence === Number(appliedPersistence);

      return classificationMatch && frpMatch && persistenceMatch;
    });
  }, [classification, appliedFRP, appliedPersistence]);

  const applyFilters = () => {
    setAppliedFRP(frpInput);
    setAppliedPersistence(persistenceInput);
    setSelectedDetection(null);
  };

  const resetFilters = () => {
    setClassification("All");

    setFrpInput("");
    setPersistenceInput("");

    setAppliedFRP("");
    setAppliedPersistence("");

    setSelectedDetection(null);
  };

  const filtersActive =
    classification !== "All" || appliedFRP !== "" || appliedPersistence !== "";

  const averageConfidence = Math.round(
    detections.reduce((total, item) => total + item.confidence, 0) /
      detections.length,
  );

  const averageFRP = (
    detections.reduce((total, item) => total + item.frp, 0) / detections.length
  ).toFixed(1);

  const priorityCount = detections.filter(
    (item) => item.risk === "High" || item.risk === "Critical",
  ).length;

  const persistentCount = detections.filter(
    (item) => item.persistence >= 4,
  ).length;

  const openOnMap = (item) => {
    resetFilters();
    setSelectedDetection(item);
    setPage("map");
  };

  const pages = [
    ["dashboard", "01", "Dashboard"],
    ["map", "02", "Detection Map"],
    ["history", "03", "Historical Analysis"],
    ["satellite", "04", "Satellite Analysis"],
  ];

  return (
    <div className="app">
      {/* ================= SIDEBAR ================= */}

      <aside className="sidebar">
        <div className="brand">
          <div className="brand-symbol">
            <span />
          </div>

          <div>
            <strong>IGNITES</strong>
            <small>THERMAL INTELLIGENCE</small>
          </div>
        </div>

        <p className="menu-title">MONITORING SYSTEM</p>

        <nav>
          {pages.map(([id, number, label]) => (
           <button
  type="button"
  onClick={(event) => {
    event.stopPropagation();
    setSelectedDetection(item);

    setTimeout(() => {
      document.querySelector(".inspector")?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }, 100);
  }}
>
  View details →
</button>
          ))}
        </nav>

        <div className="system-status">
          <div>
            <i />
            SYSTEM OPERATIONAL
          </div>

          <p>
            <span>Satellite feed</span>
            <strong>Active</strong>
          </p>

          <p>
            <span>Sources</span>
            <strong>VIIRS / MODIS</strong>
          </p>

          <p>
            <span>Records</span>
            <strong>{detections.length}</strong>
          </p>
        </div>
      </aside>

      {/* ================= MAIN ================= */}

      <div className="main">
        <header>
          <div>
            IGNITES
            <span>/</span>
            <strong>{pages.find(([id]) => id === page)?.[2]}</strong>
          </div>

          <div className="monitoring-status">
            <i />
            Monitoring Active
          </div>
        </header>

        <main className="content">
          {/* =================================================
              DASHBOARD
          ================================================= */}

          {page === "dashboard" && (
            <>
              <PageHeading
                eyebrow="OPERATIONS OVERVIEW"
                title="Thermal Monitoring Dashboard"
                description="Current thermal detections, classifications, persistence and satellite monitoring status."
              />

              <div className="stats-grid">
                <StatCard
                  title="TOTAL DETECTIONS"
                  value={detections.length}
                  subtitle="Thermal observations"
                />

                <StatCard
                  title="PRIORITY EVENTS"
                  value={priorityCount}
                  subtitle="High and critical"
                  accent
                />

                <StatCard
                  title="PERSISTENT SOURCES"
                  value={persistentCount}
                  subtitle="4 or more observations"
                />

                <StatCard
                  title="MODEL CONFIDENCE"
                  value={`${averageConfidence}%`}
                  subtitle="Average confidence"
                />
              </div>

              <div className="dashboard-grid">
                <section className="panel">
                  <div className="panel-heading">
                    <span>CLASSIFICATION</span>
                    <h2>Source Distribution</h2>
                  </div>

                  <div className="distribution">
                    {categories.map((category) => {
                      const count = detections.filter(
                        (item) => item.classification === category.name,
                      ).length;

                      return (
                        <div key={category.name}>
                          <div className="distribution-label">
                            <span>
                              <i
                                style={{
                                  background: category.color,
                                }}
                              />

                              {category.name}
                            </span>

                            <strong>{count}</strong>
                          </div>

                          <div className="progress">
                            <div
                              style={{
                                width: `${(count / detections.length) * 100}%`,
                                background: category.color,
                              }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </section>

                <section className="panel">
                  <div className="panel-heading">
                    <span>OPERATIONAL SUMMARY</span>
                    <h2>Thermal Statistics</h2>
                  </div>

                  <div className="summary-grid">
                    <div>
                      <span>Average FRP</span>
                      <strong>{averageFRP}</strong>
                      <small>MW</small>
                    </div>

                    <div>
                      <span>Maximum FRP</span>
                      <strong>
                        {Math.max(...detections.map((item) => item.frp))}
                      </strong>
                      <small>MW</small>
                    </div>

                    <div>
                      <span>VIIRS Records</span>
                      <strong>
                        {
                          detections.filter(
                            (item) => item.satellite === "VIIRS",
                          ).length
                        }
                      </strong>
                    </div>

                    <div>
                      <span>MODIS Records</span>
                      <strong>
                        {
                          detections.filter(
                            (item) => item.satellite === "MODIS",
                          ).length
                        }
                      </strong>
                    </div>
                  </div>
                </section>
              </div>

              <section className="panel">
                <div className="panel-heading row-heading">
                  <div>
                    <span>DETECTION FEED</span>
                    <h2>Recent Thermal Detections</h2>
                  </div>

                  <button
                    className="text-button"
                    onClick={() => setPage("map")}
                  >
                    Open detection map →
                  </button>
                </div>

                <div className="table-wrapper">
                  <table>
                    <thead>
                      <tr>
                        <th>ID</th>
                        <th>Classification</th>
                        <th>Location</th>
                        <th>FRP</th>
                        <th>Persistence</th>
                        <th>Confidence</th>
                        <th>Risk</th>
                        <th></th>
                      </tr>
                    </thead>

                    <tbody>
                      {detections.map((item) => (
                        <tr key={item.id}>
                          <td className="record-id">{item.id}</td>

                          <td>
                            <i
                              className="category-dot"
                              style={{
                                background: getColor(item.classification),
                              }}
                            />

                            {item.classification}
                          </td>

                          <td>{item.location}</td>

                          <td>{item.frp} MW</td>

                          <td>{item.persistence}</td>

                          <td>{item.confidence}%</td>

                          <td>
                            <RiskBadge risk={item.risk} />
                          </td>

                          <td>
                            <button
                              className="text-button"
                              onClick={() => openOnMap(item)}
                            >
                              Inspect
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            </>
          )}

          {/* =================================================
              DETECTION MAP
          ================================================= */}

          {page === "map" && (
            <>
              <PageHeading
                eyebrow="GEOSPATIAL MONITORING"
                title="Detection Map"
                description="Filter thermal anomalies using exact FRP and persistence values and inspect matching observations."
              />

              {/* EXACT FILTER PANEL */}

              <section className="filter-panel">
                <div className="filter-field classification-field">
                  <label>CLASSIFICATION</label>

                  <select
                    value={classification}
                    onChange={(event) => setClassification(event.target.value)}
                  >
                    <option value="All">All classifications</option>

                    {categories.map((item) => (
                      <option key={item.name} value={item.name}>
                        {item.name}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="filter-field">
                  <label>EXACT FRP</label>

                  <div className="input-unit">
                    <input
                      type="number"
                      step="0.1"
                      placeholder="e.g. 7.3"
                      value={frpInput}
                      onChange={(event) => setFrpInput(event.target.value)}
                    />

                    <span>MW</span>
                  </div>
                </div>

                <div className="filter-field">
                  <label>EXACT PERSISTENCE</label>

                  <div className="input-unit">
                    <input
                      type="number"
                      step="1"
                      min="0"
                      placeholder="e.g. 6"
                      value={persistenceInput}
                      onChange={(event) =>
                        setPersistenceInput(event.target.value)
                      }
                    />

                    <span>OBS</span>
                  </div>
                </div>

                <button className="apply-button" onClick={applyFilters}>
                  Apply Filter
                </button>

                <button className="reset-button" onClick={resetFilters}>
                  Reset
                </button>
              </section>

              {/* FILTER RESULT */}

              {filtersActive && (
                <section className="filter-result">
                  <div>
                    <span>FILTER RESULT</span>

                    <strong>
                      {filteredDetections.length}{" "}
                      {filteredDetections.length === 1
                        ? "thermal anomaly"
                        : "thermal anomalies"}{" "}
                      found
                    </strong>
                  </div>

                  <div className="filter-tags">
                    {classification !== "All" && (
                      <span>
                        Classification
                        <strong>{classification}</strong>
                      </span>
                    )}

                    {appliedFRP !== "" && (
                      <span>
                        Exact FRP
                        <strong>{appliedFRP} MW</strong>
                      </span>
                    )}

                    {appliedPersistence !== "" && (
                      <span>
                        Exact Persistence
                        <strong>{appliedPersistence}</strong>
                      </span>
                    )}
                  </div>
                </section>
              )}

              <div className="map-workspace">
                <div className="map-area">
                  <MapContainer
                    center={[22.9734, 78.6569]}
                    zoom={5}
                    minZoom={4}
                    maxZoom={19}
                    scrollWheelZoom
                    className="main-map"
                  >
                    <TileLayer
                      attribution="Tiles © Esri"
                      url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                      maxZoom={19}
                    />

                    <TileLayer
                      attribution="Esri"
                      url="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
                      maxZoom={19}
                    />

                    <MapFocus detection={selectedDetection} />

                    {filteredDetections.map((item) => (
                      <CircleMarker
                        key={item.id}
                        center={[item.latitude, item.longitude]}
                        radius={selectedDetection?.id === item.id ? 10 : 7}
                        pathOptions={{
                          color: "#ffffff",
                          weight: 2,
                          fillColor: getColor(item.classification),
                          fillOpacity: 0.95,
                        }}
                        eventHandlers={{
                          click: () => setSelectedDetection(item),
                        }}
                      >
                        <Popup>
                          <div className="popup popup-compact">
                            <span className="popup-eyebrow">{item.id}</span>

                            <strong>{item.classification}</strong>

                            <p>{item.location}</p>

                            <div className="popup-footer">
                              <RiskBadge risk={item.risk} />

                              <button
                                type="button"
                                onClick={() => setSelectedDetection(item)}
                              >
                                View details →
                              </button>
                            </div>
                          </div>
                        </Popup>
                      </CircleMarker>
                    ))}
                  </MapContainer>

                  <div className="visible-count">
                    <i />
                    {filteredDetections.length} visible detections
                  </div>

                  <div className="legend">
                    <strong>CLASSIFICATION</strong>

                    {categories.map((item) => (
                      <span key={item.name}>
                        <i
                          style={{
                            background: item.color,
                          }}
                        />
                        {item.name}
                      </span>
                    ))}
                  </div>
                  <SeverityLegend />
                  {filteredDetections.length === 0 && (
                    <div className="no-results">
                      <div className="target">
                        <span />
                      </div>

                      <h3>No matching thermal anomalies</h3>

                      <p>
                        No observation has the exact FRP and persistence values
                        you entered.
                      </p>

                      <button onClick={resetFilters}>Clear Filters</button>
                    </div>
                  )}
                </div>

                {/* INSPECTOR */}

                <aside className="inspector">
                  {!selectedDetection ? (
                    <div className="empty-inspector">
                      <div className="target">
                        <span />
                      </div>

                      <h3>Select an observation</h3>

                      <p>
                        Click a visible thermal anomaly on the map to inspect
                        its details.
                      </p>

                      <div className="visible-summary">
                        <span>Visible detections</span>

                        <strong>{filteredDetections.length}</strong>
                      </div>
                    </div>
                  ) : (
                    <>
                      <div className="inspection-title">
                        <span>SELECTED DETECTION</span>

                        <div className="inspection-heading-row">
                          <div>
                            <h2>{selectedDetection.classification}</h2>
                            <p>{selectedDetection.location}</p>
                          </div>

                          <RiskBadge risk={selectedDetection.risk} />
                        </div>

                        <div className="inspection-meta">
                          <strong>{selectedDetection.id}</strong>
                          <span>
                            {selectedDetection.confidence}% model confidence
                          </span>
                        </div>
                      </div>

                      <ClassificationEvidence detection={selectedDetection} />

                      <div className="details">
                        <h4>OBSERVATION</h4>

                        <DetailRow
                          label="Location"
                          value={selectedDetection.location}
                        />

                        <DetailRow
                          label="Latitude"
                          value={selectedDetection.latitude}
                        />

                        <DetailRow
                          label="Longitude"
                          value={selectedDetection.longitude}
                        />

                        <DetailRow
                          label="Satellite"
                          value={selectedDetection.satellite}
                        />

                        <DetailRow
                          label="Timestamp"
                          value={selectedDetection.timestamp}
                        />
                      </div>

                      <div className="details">
                        <h4>THERMAL MEASUREMENTS</h4>

                        <DetailRow
                          label="Brightness"
                          value={`${selectedDetection.brightness} K`}
                        />

                        <DetailRow
                          label="FRP"
                          value={`${selectedDetection.frp} MW`}
                        />

                        <DetailRow
                          label="Persistence"
                          value={`${selectedDetection.persistence} observations`}
                        />

                        <DetailRow
                          label="Confidence"
                          value={`${selectedDetection.confidence}%`}
                        />
                      </div>
                      <PersistenceTimeline detection={selectedDetection} />
                      <OSMContext detection={selectedDetection} />
                      <div className="details">
                        <h4>SPATIAL CONTEXT</h4>

                        <DetailRow
                          label="Land cover"
                          value={selectedDetection.landCover}
                        />

                        <DetailRow
                          label="Nearest feature"
                          value={selectedDetection.nearestFeature}
                        />

                        <DetailRow
                          label="Distance"
                          value={`${selectedDetection.distance} km`}
                        />
                      </div>

                      <button
                        className="satellite-button"
                        onClick={() => setPage("satellite")}
                      >
                        Open Satellite Analysis
                      </button>
                    </>
                  )}
                </aside>
              </div>
            </>
          )}

          {/* =================================================
              HISTORY
          ================================================= */}

          {page === "history" && (
            <>
              <PageHeading
                eyebrow="TEMPORAL INTELLIGENCE"
                title="Historical Analysis"
                description="Analyse recurring thermal behaviour, persistence and historical fire radiative power."
              />

              <div className="stats-grid">
                <StatCard
                  title="OBSERVATIONS"
                  value={detections.length}
                  subtitle="Historical records"
                />

                <StatCard
                  title="PERSISTENT SOURCES"
                  value={persistentCount}
                  subtitle="Recurring observations"
                  accent
                />

                <StatCard
                  title="AVERAGE FRP"
                  value={averageFRP}
                  subtitle="MW"
                />

                <StatCard
                  title="CONFIDENCE"
                  value={`${averageConfidence}%`}
                  subtitle="Average model confidence"
                />
              </div>

              <div className="dashboard-grid">
                <section className="panel">
                  <div className="panel-heading">
                    <span>RECURRING SOURCES</span>

                    <h2>Persistence Analysis</h2>
                  </div>

                  <div className="bar-list">
                    {[...detections]
                      .sort((a, b) => b.persistence - a.persistence)
                      .map((item) => (
                        <div key={item.id}>
                          <span>{item.id}</span>

                          <div>
                            <i
                              style={{
                                width: `${item.persistence * 10}%`,
                                background: getColor(item.classification),
                              }}
                            />
                          </div>

                          <strong>{item.persistence}</strong>
                        </div>
                      ))}
                  </div>
                </section>

                <section className="panel">
                  <div className="panel-heading">
                    <span>THERMAL ENERGY</span>

                    <h2>FRP Comparison</h2>
                  </div>

                  <div className="bar-list">
                    {[...detections]
                      .sort((a, b) => b.frp - a.frp)
                      .map((item) => (
                        <div key={item.id}>
                          <span>{item.id}</span>

                          <div>
                            <i
                              style={{
                                width: `${(item.frp / 35) * 100}%`,
                              }}
                            />
                          </div>

                          <strong>{item.frp}</strong>
                        </div>
                      ))}
                  </div>
                </section>
              </div>

              <section className="panel">
                <div className="panel-heading">
                  <span>DETECTION ARCHIVE</span>
                  <h2>Historical Records</h2>
                </div>

                <div className="table-wrapper">
                  <table>
                    <thead>
                      <tr>
                        <th>ID</th>
                        <th>Timestamp</th>
                        <th>Classification</th>
                        <th>Location</th>
                        <th>FRP</th>
                        <th>Persistence</th>
                        <th>Confidence</th>
                        <th>Risk</th>
                      </tr>
                    </thead>

                    <tbody>
                      {detections.map((item) => (
                        <tr
                          key={item.id}
                          onClick={() => openOnMap(item)}
                          className="clickable"
                        >
                          <td className="record-id">{item.id}</td>

                          <td>{item.timestamp}</td>

                          <td>{item.classification}</td>

                          <td>{item.location}</td>

                          <td>{item.frp} MW</td>

                          <td>{item.persistence}</td>

                          <td>{item.confidence}%</td>

                          <td>
                            <RiskBadge risk={item.risk} />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            </>
          )}

          {/* =================================================
              SATELLITE ANALYSIS
          ================================================= */}

          {page === "satellite" && (
            <>
              <PageHeading
                eyebrow="REMOTE SENSING WORKSPACE"
                title="Satellite Analysis"
                description="Inspect satellite imagery, thermal measurements, spectral indices and surrounding spatial context."
              />

              <div className="satellite-select">
                <label>SELECT DETECTION</label>

                <select
                  value={selectedDetection?.id || ""}
                  onChange={(event) => {
                    const item = detections.find(
                      (detection) => detection.id === event.target.value,
                    );

                    setSelectedDetection(item || null);
                  }}
                >
                  <option value="">Select observation</option>

                  {detections.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.id} — {item.classification}
                    </option>
                  ))}
                </select>
              </div>

              {!selectedDetection ? (
                <section className="satellite-empty">
                  <div className="target">
                    <span />
                  </div>

                  <h2>No observation selected</h2>

                  <p>
                    Select a detection to inspect satellite imagery and spectral
                    information.
                  </p>

                  <button
                    className="satellite-button"
                    onClick={() => setPage("map")}
                  >
                    Open Detection Map
                  </button>
                </section>
              ) : (
                <>
                  <section className="analysis-header">
                    <div>
                      <span>ACTIVE OBSERVATION</span>

                      <h2>{selectedDetection.classification}</h2>

                      <p>
                        {selectedDetection.id} · {selectedDetection.location}
                      </p>
                    </div>

                    <RiskBadge risk={selectedDetection.risk} />
                  </section>

                  <div className="spectral-grid">
                    <StatCard
                      title="NDVI"
                      value={selectedDetection.ndvi}
                      subtitle="Vegetation index"
                    />

                    <StatCard
                      title="NBR"
                      value={selectedDetection.nbr}
                      subtitle="Normalized burn ratio"
                    />

                    <StatCard
                      title="dNBR"
                      value={selectedDetection.dnbr}
                      subtitle="Burn severity"
                    />

                    <StatCard
                      title="FRP"
                      value={`${selectedDetection.frp}`}
                      subtitle="MW"
                      accent
                    />
                  </div>

                  <div className="satellite-grid">
                    <section className="panel">
                      <div className="panel-heading">
                        <span>SATELLITE IMAGERY</span>

                        <h2>Observation Context</h2>
                      </div>

                      <MapContainer
                        key={selectedDetection.id}
                        center={[
                          selectedDetection.latitude,
                          selectedDetection.longitude,
                        ]}
                        zoom={16}
                        maxZoom={19}
                        className="satellite-map"
                      >
                        <TileLayer
                          attribution="Tiles © Esri"
                          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                          maxZoom={19}
                        />

                        <TileLayer
                          attribution="Esri"
                          url="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
                          maxZoom={19}
                        />

                        <CircleMarker
                          center={[
                            selectedDetection.latitude,
                            selectedDetection.longitude,
                          ]}
                          radius={9}
                          pathOptions={{
                            color: "#ffffff",
                            weight: 2,
                            fillColor: getColor(
                              selectedDetection.classification,
                            ),
                            fillOpacity: 1,
                          }}
                        >
                          <Popup>{selectedDetection.classification}</Popup>
                        </CircleMarker>
                      </MapContainer>
                    </section>

                    <section className="panel">
                      <div className="panel-heading">
                        <span>ANALYSIS</span>

                        <h2>Observation Properties</h2>
                      </div>

                      <div className="details">
                        <h4>THERMAL</h4>

                        <DetailRow
                          label="Brightness"
                          value={`${selectedDetection.brightness} K`}
                        />

                        <DetailRow
                          label="FRP"
                          value={`${selectedDetection.frp} MW`}
                        />

                        <DetailRow
                          label="Persistence"
                          value={selectedDetection.persistence}
                        />
                      </div>

                      <div className="details">
                        <h4>SPECTRAL</h4>

                        <DetailRow
                          label="NDVI"
                          value={selectedDetection.ndvi}
                        />

                        <DetailRow label="NBR" value={selectedDetection.nbr} />

                        <DetailRow
                          label="dNBR"
                          value={selectedDetection.dnbr}
                        />
                      </div>

                      <div className="details">
                        <h4>SPATIAL CONTEXT</h4>

                        <DetailRow
                          label="Land cover"
                          value={selectedDetection.landCover}
                        />

                        <DetailRow
                          label="Nearest feature"
                          value={selectedDetection.nearestFeature}
                        />

                        <DetailRow
                          label="Distance"
                          value={`${selectedDetection.distance} km`}
                        />
                      </div>
                    </section>
                  </div>
                </>
              )}
            </>
          )}
        </main>

        <footer>
          <span>IGNITES · Industrial Fire & Thermal Source Monitoring</span>

          <span>SIH26162 · Development Prototype</span>
        </footer>
      </div>
    </div>
  );
}
