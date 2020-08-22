var app = getApp()

// 1、引入依赖脚本
import * as echarts from '../../ec-canvas/echarts';

let chart = null;
let radar = null;

// 2、进行初始化数据
function initChart(canvas, width, height) {
  chart = echarts.init(canvas, null, {
    width: width,
    height: height
  });
  canvas.setChart(chart);
  var option = {
    color: ["#3398DB"],
    tooltip: {
      trigger: "axis",
      axisPointer: {
        type: "shadow"
      }
    },
    legend: {
      data: ['评分']
    },
    grid: {
      height:350
    },
    xAxis: [{
      type: "category",
      data: ['位置', '服务', '价格', '环境', '菜品', '其他'],
    }],
    yAxis: [{
      type: "value",
      max: 100
    }],
    series: [{
      name: "评分",
      type: "bar",
      barWidth: "60%",
      data: app.globalData.scoreList
    }]
  }

  chart.setOption(option);
  return chart;
}

function initRadar(canvas, width, height, dpr) {
  const radar = echarts.init(canvas, null, {
    width: width,
    height: height,
    devicePixelRatio: dpr // new
  });
  canvas.setChart(radar);

  var option = {
    backgroundColor: "#ffffff",
    color: ["#37A2DA"],
    xAxis: {
      show: false
    },
    yAxis: {
      show: false
    },
    // legend: {
    //   data: ["Aspect Radar Score"]
    // },
    radar: {
      // shape: 'circle',
      indicator: [{
        name: '交通便利',
        max: 300
      },
      {
        name: '距离商圈',
        max: 300
      },
      {
        name: '容易寻找',
        max: 300
      },
      {
        name: '排队时间',
        max: 300
      },
      {
        name: '服务态度',
        max: 300
      },
      {
        name: '容易停车',
        max: 300
      },
      {
        name: '上菜速度',
        max: 300
      },
      {
        name: '价格水平',
        max: 300
      },
      {
        name: '性价比',
        max: 300
      },
      {
        name: '折扣力度',
        max: 300
      },
      {
        name: '装修情况',
        max: 300
      },
      {
        name: '嘈杂情况',
        max: 300
      },
      {
        name: '就餐空间',
        max: 300
      },
      {
        name: '卫生情况',
        max: 300
      },
      {
        name: '菜品分量',
        max: 300
      },
      {
        name: '菜品口感',
        max: 300
      },
      {
        name: '菜品外观',
        max: 300
      },
      {
        name: '菜品推荐程度',
        max: 300
      },
      {
        name: '本次消费感受',
        max: 300
      },
      {
        name: '再次消费意愿',
        max: 300
      }
      ],
      radius: ["25%", "75%"],
    },
    series: [{
      name: 'Aspect Radar Score',
      type: 'radar',
      areaStyle: {
        color: '#FFB6C1',
        // opacity: 0.6  // 透明度
      },
      data: [{
        value: app.globalData.aspectScore,
        name: 'Aspect Radar Score'
      }
      ]
    }]
  };

  radar.setOption(option);
  return radar;
}

Page({

  data: {
    ec: {
      onInit: initChart // 3、将数据放入到里面
    },

    ec1: {
      onInit: initRadar // 3、将数据放入到里面
    }
  },
  onReady() {
    setTimeout(function () {
      // 获取 chart 实例的方式
      console.log(chart)
      console.log(radar)
    }, 2000);
  }
});

