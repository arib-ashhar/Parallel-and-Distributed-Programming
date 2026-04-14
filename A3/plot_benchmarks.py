#!/usr/bin/env python3

import csv
import math
from pathlib import Path


class BenchmarkPoint:
    def __init__(self, processes, executionMs, wallTimeSeconds, speedup, efficiency, profit):
        self.Processes = processes
        self.ExecutionMs = executionMs
        self.WallTimeSeconds = wallTimeSeconds
        self.Speedup = speedup
        self.Efficiency = efficiency
        self.Profit = profit


class SvgChartBuilder:
    def __init__(self, width, height, marginLeft, marginRight, marginTop, marginBottom):
        self.Width = width
        self.Height = height
        self.MarginLeft = marginLeft
        self.MarginRight = marginRight
        self.MarginTop = marginTop
        self.MarginBottom = marginBottom
        self.PlotWidth = width - marginLeft - marginRight
        self.PlotHeight = height - marginTop - marginBottom

    # Maps an x value to canvas space.
    def MapX(self, xValue, xMin, xMax):
        if xMax == xMin:
            return self.MarginLeft + self.PlotWidth / 2.0
        return self.MarginLeft + ((xValue - xMin) / (xMax - xMin)) * self.PlotWidth

    # Maps a y value to canvas space.
    def MapY(self, yValue, yMin, yMax):
        if yMax == yMin:
            return self.MarginTop + self.PlotHeight / 2.0
        return self.MarginTop + self.PlotHeight - ((yValue - yMin) / (yMax - yMin)) * self.PlotHeight


# Reads benchmark rows from CSV.
def LoadBenchmarkData(csvPath):
    points = []
    with open(csvPath, newline="") as csvFile:
        reader = csv.DictReader(csvFile)
        for row in reader:
            points.append(
                BenchmarkPoint(
                    int(row["Processes"]),
                    int(row["ExecutionMs"]),
                    float(row["WallTimeSeconds"]),
                    float(row["Speedup"]),
                    float(row["Efficiency"]),
                    int(row["Profit"]),
                )
            )
    return points


# Produces a nice upper bound for the y axis.
def ComputeYAxisLimit(values):
    maximumValue = max(values)
    if maximumValue <= 1.0:
        return 1.0

    exponent = math.floor(math.log10(maximumValue))
    base = 10 ** exponent
    normalized = maximumValue / base

    if normalized <= 1.0:
        niceNormalized = 1.0
    elif normalized <= 2.0:
        niceNormalized = 2.0
    elif normalized <= 5.0:
        niceNormalized = 5.0
    else:
        niceNormalized = 10.0

    return niceNormalized * base


# Builds y tick positions and labels.
def BuildTicks(yMax, tickCount):
    ticks = []
    for tickIndex in range(tickCount + 1):
        value = (yMax * tickIndex) / tickCount
        if yMax <= 2.0:
            label = f"{value:.2f}"
        elif yMax <= 20.0:
            label = f"{value:.1f}"
        else:
            label = f"{int(round(value))}"
        ticks.append((value, label))
    return ticks


# Renders a single line chart SVG.
def RenderLineChart(points, title, yLabel, valueAccessor, outputPath, yMaxOverride=None, idealLineAccessor=None):
    builder = SvgChartBuilder(900, 560, 95, 40, 70, 85)
    xValues = [point.Processes for point in points]
    yValues = [valueAccessor(point) for point in points]
    xMin = min(xValues)
    xMax = max(xValues)
    yMin = 0.0
    yMax = yMaxOverride if yMaxOverride is not None else ComputeYAxisLimit(yValues)

    tickPairs = BuildTicks(yMax, 5)
    polylinePoints = []
    for point in points:
        mappedX = builder.MapX(point.Processes, xMin, xMax)
        mappedY = builder.MapY(valueAccessor(point), yMin, yMax)
        polylinePoints.append(f"{mappedX:.2f},{mappedY:.2f}")

    svgLines = []
    svgLines.append('<?xml version="1.0" encoding="UTF-8"?>')
    svgLines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{builder.Width}" height="{builder.Height}" viewBox="0 0 {builder.Width} {builder.Height}">'
    )
    svgLines.append('<rect width="100%" height="100%" fill="#fffdf8"/>')
    svgLines.append(f'<text x="{builder.Width / 2:.2f}" y="34" text-anchor="middle" font-size="26" font-family="Helvetica, Arial, sans-serif" fill="#1f2937">{title}</text>')

    for tickValue, tickLabel in tickPairs:
        mappedY = builder.MapY(tickValue, yMin, yMax)
        svgLines.append(
            f'<line x1="{builder.MarginLeft}" y1="{mappedY:.2f}" x2="{builder.Width - builder.MarginRight}" y2="{mappedY:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        svgLines.append(
            f'<text x="{builder.MarginLeft - 12}" y="{mappedY + 5:.2f}" text-anchor="end" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#4b5563">{tickLabel}</text>'
        )

    for point in points:
        mappedX = builder.MapX(point.Processes, xMin, xMax)
        svgLines.append(
            f'<line x1="{mappedX:.2f}" y1="{builder.MarginTop}" x2="{mappedX:.2f}" y2="{builder.Height - builder.MarginBottom}" stroke="#f1f5f9" stroke-width="1"/>'
        )
        svgLines.append(
            f'<text x="{mappedX:.2f}" y="{builder.Height - builder.MarginBottom + 28}" text-anchor="middle" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#4b5563">{point.Processes}</text>'
        )

    svgLines.append(
        f'<line x1="{builder.MarginLeft}" y1="{builder.Height - builder.MarginBottom}" x2="{builder.Width - builder.MarginRight}" y2="{builder.Height - builder.MarginBottom}" stroke="#111827" stroke-width="2"/>'
    )
    svgLines.append(
        f'<line x1="{builder.MarginLeft}" y1="{builder.MarginTop}" x2="{builder.MarginLeft}" y2="{builder.Height - builder.MarginBottom}" stroke="#111827" stroke-width="2"/>'
    )

    if idealLineAccessor is not None:
        idealPolylinePoints = []
        for point in points:
            mappedX = builder.MapX(point.Processes, xMin, xMax)
            mappedY = builder.MapY(idealLineAccessor(point), yMin, yMax)
            idealPolylinePoints.append(f"{mappedX:.2f},{mappedY:.2f}")
        svgLines.append(
            f'<polyline fill="none" stroke="#f59e0b" stroke-width="3" stroke-dasharray="10 6" points="{" ".join(idealPolylinePoints)}"/>'
        )

    svgLines.append(
        f'<polyline fill="none" stroke="#0f766e" stroke-width="4" points="{" ".join(polylinePoints)}"/>'
    )

    for point in points:
        mappedX = builder.MapX(point.Processes, xMin, xMax)
        mappedY = builder.MapY(valueAccessor(point), yMin, yMax)
        svgLines.append(f'<circle cx="{mappedX:.2f}" cy="{mappedY:.2f}" r="6" fill="#0f766e"/>')
        svgLines.append(
            f'<text x="{mappedX:.2f}" y="{mappedY - 14:.2f}" text-anchor="middle" font-size="13" font-family="Helvetica, Arial, sans-serif" fill="#111827">{valueAccessor(point):.3g}</text>'
        )

    svgLines.append(
        f'<text x="{builder.Width / 2:.2f}" y="{builder.Height - 20}" text-anchor="middle" font-size="18" font-family="Helvetica, Arial, sans-serif" fill="#1f2937">Number of MPI Processes</text>'
    )

    svgLines.append(
        f'<text x="28" y="{builder.Height / 2:.2f}" text-anchor="middle" font-size="18" font-family="Helvetica, Arial, sans-serif" fill="#1f2937" transform="rotate(-90 28 {builder.Height / 2:.2f})">{yLabel}</text>'
    )

    if idealLineAccessor is not None:
        legendX = builder.Width - 255
        legendY = builder.MarginTop + 10
        svgLines.append(f'<rect x="{legendX}" y="{legendY}" width="205" height="62" rx="8" fill="#ffffff" stroke="#d1d5db"/>')
        svgLines.append(f'<line x1="{legendX + 16}" y1="{legendY + 20}" x2="{legendX + 52}" y2="{legendY + 20}" stroke="#0f766e" stroke-width="4"/>')
        svgLines.append(f'<text x="{legendX + 62}" y="{legendY + 25}" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#111827">Measured</text>')
        svgLines.append(f'<line x1="{legendX + 16}" y1="{legendY + 44}" x2="{legendX + 52}" y2="{legendY + 44}" stroke="#f59e0b" stroke-width="3" stroke-dasharray="10 6"/>')
        svgLines.append(f'<text x="{legendX + 62}" y="{legendY + 49}" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#111827">Ideal</text>')

    svgLines.append("</svg>")
    Path(outputPath).write_text("\n".join(svgLines), encoding="utf-8")


# Generates all benchmark figures.
def Main():
    rootDir = Path(__file__).resolve().parent
    csvPath = rootDir / "benchmark_data.csv"
    outputDir = rootDir / "figures"
    outputDir.mkdir(exist_ok=True)

    points = LoadBenchmarkData(csvPath)
    baseExecutionMs = points[0].ExecutionMs

    RenderLineChart(
        points,
        "Execution Time vs MPI Processes",
        "Execution Time (ms)",
        lambda point: point.ExecutionMs,
        outputDir / "execution_time.svg",
    )

    RenderLineChart(
        points,
        "Speedup vs MPI Processes",
        "Speedup",
        lambda point: point.Speedup,
        outputDir / "speedup.svg",
        yMaxOverride=16.0,
        idealLineAccessor=lambda point: float(point.Processes),
    )

    RenderLineChart(
        points,
        "Parallel Efficiency vs MPI Processes",
        "Efficiency",
        lambda point: point.Efficiency,
        outputDir / "efficiency.svg",
        yMaxOverride=1.0,
    )

    summaryPath = outputDir / "summary.txt"
    summaryPath.write_text(
        "\n".join(
            [
                f"Base execution time: {baseExecutionMs} ms",
                f"Best wall-clock time: {min(point.WallTimeSeconds for point in points):.2f} s",
                f"Best speedup: {max(point.Speedup for point in points):.3f}",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    Main()
