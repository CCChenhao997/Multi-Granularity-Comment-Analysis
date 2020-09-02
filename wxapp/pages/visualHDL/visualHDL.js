var app = getApp()

// 1、引入依赖脚本
import * as echarts from '../../ec-canvas/echarts';

let chart = null;
let bar = null;

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
      data: app.globalData.hdlScoreList
    }]
  }

  chart.setOption(option);
  return chart;
}

function initBar(canvas, width, height, dpr) {
  bar = echarts.init(canvas, null, {
    width: width,
    height: height,
    devicePixelRatio: dpr // new
  });
  canvas.setChart(bar);

  var option = {
    color: ['#37a2da', '#32c5e9', '#67e0e3'],
    tooltip: {
      trigger: 'axis',
      axisPointer: {            // 坐标轴指示器，坐标轴触发有效
        type: 'shadow'        // 默认为直线，可选为：'line' | 'shadow'
      },
      confine: true
    },
    legend: {
      data: ['正面', '中性', '负面']
    },
    grid: {
      left: 20,
      right: 20,
      bottom: 15,
      top: 40,
      containLabel: true
    },
    xAxis: [
      {
        type: 'value',
        axisLine: {
          lineStyle: {
            color: '#999'
          }
        },
        axisLabel: {
          color: '#666'
        }
      }
    ],
    yAxis: [
      {
        type: 'category',
        axisTick: { show: false },
        // data: ['汽车之家', '今日头条', '百度贴吧', '一点资讯', '微信', '微博', '知乎'],
        // data: ['交通便利', '距离商圈', '容易寻找', '排队时间', '服务态度', 
        //        '容易停车', '上菜速度', '价格水平', '性价比', '折扣力度', 
        //        '装修情况', '嘈杂情况', '就餐空间', '卫生情况', '菜品分量',
        //        '菜品口感', '菜品外观', '菜品推荐程度', '本次消费感受', '再次消费意愿'],
        
        data: ['再次消费意愿', '本次消费感受', '菜品推荐程度', '菜品外观', '菜品口感', 
               '菜品分量', '卫生情况', '就餐空间', '嘈杂情况', '装修情况', 
               '折扣力度', '性价比', '价格水平', '上菜速度', '容易停车',
               '服务态度', '排队时间', '容易寻找', '距离商圈', '交通便利'],       
        
        axisLine: {
          lineStyle: {
            color: '#999'
          }
        },
        axisLabel: {
          color: '#666'
        }
      }
    ],
    series: [
      {
        name: '正面',
        type: 'bar',
        label: {
          normal: {
            show: true,
            position: 'inside'
          }
        },
        // data: [300, 270, 340, 344, 300, 320, 310],
        data: app.globalData.hdlAspectPositiveScoreList,
        itemStyle: {
          // emphasis: {
          //   color: '#37a2da'
          // }
        }
      },
      {
        name: '中性',
        type: 'bar',
        stack: '总量',
        label: {
          normal: {
            show: true,
            position: 'right'
          }
        },
        // data: [120, 102, 141, 174, 190, 250, 220],
        data: app.globalData.hdlAspectNeutralScoreList,
        itemStyle: {
          // emphasis: {
          //   color: '#32c5e9'
          // }
        }
      },
      {
        name: '负面',
        type: 'bar',
        stack: '总量',
        label: {
          normal: {
            show: true,
            position: 'left'
          }
        },
        // data: [-20, -32, -21, -34, -90, -130, -110],
        data: app.globalData.hdlAspectNegativeScoreList,
        itemStyle: {
          // emphasis: {
          //   color: '#67e0e3'
          // }
        }
      }
    ]
  };

  bar.setOption(option);
  return bar;
}


Page({

  data: {
    ec: {
      onInit: initChart
    },

    ec1: {
      onInit: initBar
    }
  },
  onReady() {
    setTimeout(function () {
      // 获取 chart 实例的方式
      console.log(chart)
      console.log(bar)
    }, 2000);
  }
});

