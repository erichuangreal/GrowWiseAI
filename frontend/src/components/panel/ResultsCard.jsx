// src/components/panel/ResultsCard.jsx

function clamp01(x) {
  if (typeof x !== "number") return 0;
  return Math.max(0, Math.min(1, x));
}

export default function ResultsCard({ result, isLoading }) {
  const hasResult = !!result;
  const survivability = clamp01(result?.survivability);
  const factors = Array.isArray(result?.key_factors) ? result.key_factors : [];

  const percent = Math.round(survivability * 100);

  const getCategory = (p) => {
    if (!hasResult) return { label: "Unknown", color: "#9aa0a6" };
    if (p < 25) return { label: "Not Optimal", color: "#e74c3c" };
    if (p < 50) return { label: "Sub Optimal", color: "#e0cf38ff" };
    if (p < 75) return { label: "Optimal", color: "#55eb30ff" };
    return { label: "Very Optimal", color: "#04fde4ff" };
  };

  const category = getCategory(percent);

  return (
    <div style={styles.card}>
      {/* Header */}
      <div style={styles.header}>
        <div
          style={{
            ...styles.badge,
            borderColor: category.color,
            color: category.color,
            background: `${category.color}22`,
          }}
        >
          {category.label}
        </div>

        <div style={{ flex: 1 }}>
          <div style={styles.labelRow}>
            <span style={styles.label}>Growth Conditions</span>
          </div>
        </div>
      </div>

      {isLoading ? (
        <div style={styles.loading}>Generating prediction…</div>
      ) : (
        <>
          {factors.length > 0 && (
            <div style={styles.section}>
              <div style={styles.sectionTitle}>Key factors</div>
              <ul style={styles.list}>
                {factors.slice(0, 5).map((f, i) => (
                  <li key={i} style={styles.listItem}>
                    {f}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div style={styles.meterBottom}>
            <div style={styles.meterTrack}>
              <div
                style={{
                  ...styles.meterFill,
                  width: hasResult ? `${percent}%` : "0%",
                  background: category.color,
                }}
              />
            </div>
            <div style={styles.meterText}>
              Survivability: <b>{hasResult ? `${percent}%` : "—"}</b>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

const styles = {
  card: {
    border: "1px solid #123322",
    borderRadius: 14,
    padding: "12px 12px 18px 12px",
    background: "rgba(0,0,0,0.35)",
    color: "#fff",
  },

  header: {
    display: "flex",
    gap: 12,
    alignItems: "flex-start",
  },

  badge: {
    padding: "6px 10px",
    borderRadius: 999,
    margin: "2px 0px",
    border: "1px solid",
    fontSize: 11,
    fontWeight: 900,
    letterSpacing: 0.6,
  },

  labelRow: {
    display: "flex",
    alignItems: "baseline",
    justifyContent: "space-between",
    gap: 10,
  },

  label: {
    fontSize: 16,
    fontWeight: 900,
    display: "inline-block",
    marginTop: 4,
  },

  section: {
    marginTop: 1,
    borderTop: "1px solid rgba(255,255,255,0.10)",
    paddingTop: 12,
  },
  sectionTitle: {
    fontSize: 12,
    fontWeight: 900,
    letterSpacing: 0.3,
    color: "rgba(255,255,255,0.8)",
  },

  list: { margin: "8px 0 0", paddingLeft: 18 },
  listItem: {
    marginBottom: 6,
    color: "rgba(255,255,255,0.85)",
    fontSize: 13,
  },

  loading: {
    marginTop: 12,
    padding: 10,
    borderRadius: 12,
    border: "1px solid rgba(255,255,255,0.12)",
    background: "rgba(0,0,0,0.25)",
    color: "rgba(255,255,255,0.85)",
    fontSize: 13,
  },

  hint: {
    marginTop: 12,
    padding: 10,
    borderRadius: 12,
    border: "1px dashed rgba(255,255,255,0.18)",
    background: "rgba(0,0,0,0.20)",
    color: "rgba(255,255,255,0.70)",
    fontSize: 13,
  },

  meterBottom: { marginTop: 14 },
  meterTrack: {
    width: "100%",
    height: 14,
    borderRadius: 999,
    background: "rgba(255,255,255,0.08)",
    overflow: "hidden",
    border: "1px solid rgba(255,255,255,0.10)",
  },
  meterFill: {
    height: "100%",
    borderRadius: 999,
  },
  meterText: {
    marginTop: 6,
    fontSize: 13,
    color: "rgba(255,255,255,0.85)",
    textAlign: "center",
  },
};
