import matplotlib.pyplot as plt
import base64
from io import BytesIO

# 🔥 한글 폰트 설정 유지
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def generate_base64_graph(data: dict):
    years = sorted(data.keys())
    values = [data[y] for y in years]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)

    # ✅ 완전 투명 배경 (검정 UI 위에 얹기)
    fig.patch.set_alpha(0)
    ax.set_facecolor((0, 0, 0, 0))

    # ✅ 선 색상 밝게
    ax.plot(
        years,
        values,
        marker='o',
        linewidth=2.6,
        markersize=6,
        color="#8BDAFF"   
    )

    # ✅ 제목 / 축 글자 색상 밝게
    # ax.set_title("경쟁률", fontsize=14, fontweight="bold", pad=12, color="white")
    # ax.set_xlabel("연도", fontsize=12, color="white")
    # ax.set_ylabel("경쟁률", fontsize=12, color="white")

    # ✅ 눈금 색상
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    ax.set_xticks(years)
    ax.set_yticks(range(0, int(max(values)) + 2))

    # ✅ 그리드도 밝게
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.3, color="white")

    # ✅ 테두리 색상
    for spine in ax.spines.values():
        spine.set_color("white")

    # ✅ 값 라벨 (가장 중요)
    for x, y in zip(years, values):
        ax.annotate(
            f"{y:g}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 11),
            ha="center",
            fontsize=11,
            color="white"
        )

    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        transparent=True
    )
    plt.close()
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")
