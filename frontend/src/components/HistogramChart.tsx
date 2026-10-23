import React, { useMemo } from 'react';
import {
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  Title,
  Tooltip,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

interface HistogramChartProps {
  samples: number[];
  baselineMm: number;
  thresholdMm: number;
  p05: number;
  p50: number;
  p95: number;
}

export const HistogramChart: React.FC<HistogramChartProps> = ({
  samples,
  baselineMm,
  thresholdMm,
  p05,
  p50,
  p95,
}) => {
  const chartData = useMemo(() => {
    if (!samples || samples.length === 0) {
      return { labels: [], datasets: [] };
    }

    const min = Math.min(...samples);
    const max = Math.max(...samples);
    const numBins = Math.min(20, Math.max(10, Math.round(Math.sqrt(samples.length))));
    const binWidth = (max - min) / numBins || 0.1;

    const binCounts = new Array(numBins).fill(0);
    const binLabels: string[] = [];
    const binColors: string[] = [];

    for (let i = 0; i < numBins; i++) {
      const binStart = min + i * binWidth;
      const binEnd = binStart + binWidth;
      binLabels.push(`${binStart.toFixed(2)} - ${binEnd.toFixed(2)}`);

      // Color bins below threshold in warning amber/coral
      if (binEnd <= thresholdMm) {
        binColors.push('rgba(244, 63, 94, 0.7)'); // Below threshold
      } else if (binStart < thresholdMm) {
        binColors.push('rgba(251, 191, 36, 0.7)'); // Spans threshold
      } else {
        binColors.push('rgba(56, 189, 248, 0.7)'); // Safe margin
      }
    }

    for (const val of samples) {
      let idx = Math.floor((val - min) / binWidth);
      if (idx >= numBins) idx = numBins - 1;
      if (idx < 0) idx = 0;
      binCounts[idx]++;
    }

    return {
      labels: binLabels,
      datasets: [
        {
          label: 'Realization Count',
          data: binCounts,
          backgroundColor: binColors,
          borderColor: binColors.map((c) => c.replace('0.7', '1.0')),
          borderWidth: 1,
          borderRadius: 3,
        },
      ],
    };
  }, [samples, thresholdMm]);

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        backgroundColor: '#0f172a',
        borderColor: '#334155',
        borderWidth: 1,
        titleColor: '#f8fafc',
        bodyColor: '#cbd5e1',
        callbacks: {
          title: (items: { label?: string }[]) => `Range: ${items[0]?.label || ''} mm`,
          label: (item: { raw?: unknown }) => `Realizations: ${item.raw}`,
        },
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Clearance Range (mm)',
          color: '#94a3b8',
          font: { size: 11 },
        },
        grid: {
          color: 'rgba(51, 65, 85, 0.3)',
        },
        ticks: {
          color: '#64748b',
          maxRotation: 45,
          minRotation: 45,
          font: { size: 10 },
        },
      },
      y: {
        title: {
          display: true,
          text: 'Frequency',
          color: '#94a3b8',
          font: { size: 11 },
        },
        grid: {
          color: 'rgba(51, 65, 85, 0.3)',
        },
        ticks: {
          color: '#64748b',
          precision: 0,
          font: { size: 10 },
        },
      },
    },
  };

  return (
    <div className="histogram-container">
      <div className="histogram-header">
        <span className="histogram-title">Monte Carlo Clearance Distribution</span>
        <div className="histogram-legend-pills">
          <span className="pill pill-baseline">Baseline: {baselineMm.toFixed(2)} mm</span>
          <span className="pill pill-p50">Median (p50): {p50.toFixed(2)} mm</span>
          <span className="pill pill-ci">90% CI: [{p05.toFixed(2)}, {p95.toFixed(2)}] mm</span>
          <span className="pill pill-thresh">Threshold: {thresholdMm.toFixed(2)} mm</span>
        </div>
      </div>
      <div className="chart-wrapper">
        <Bar data={chartData} options={options} />
      </div>
    </div>
  );
};
