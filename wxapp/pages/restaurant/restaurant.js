//index.js
//获取应用实例
var app = getApp();
var cardTeams;
Page({
  data: {
    stars: [0, 1, 2, 3, 4],
    normalSrc: '../../source/1.1.png',
    selectedSrc: '../../source/1.3.png',
    halfSrc: '../../source/1.2.png',
    cardTeams: [{
      "viewid": "1",
      "imgsrc": "../../source/haidilao.jpeg",
      "name": "海底捞火锅",
      "url": "../visualHDL/visualHDL",
      "score": 0
    }, 
    {
      "viewid": "2",
      "imgsrc": "../../source/kaorou.jpeg",
      "name": "新石器烤肉",
      "url": "../visualXSQ/visualXSQ",
      "score": 0
    }, 
    {
      "viewid": "3",
      "imgsrc": "../../source/tanyu.jpeg",
      "name": "探鱼",
      "url": "../visualTY/visualTY",
      "score": 0
    },
    {
      "viewid": "4",
      "imgsrc": "../../source/dapaidang.jpeg",
      "name": "南京大排档",
      "url": "../visualNJ/visualNJ",
      "score": 0
    },
    {
      "viewid": "5",
      "imgsrc": "../../source/duolunduo.jpeg",
      "name": "多伦多海鲜自助",
      "url": "../visualDLD/visualDLD",
      "score": 0
    }],
  },
  onLoad: function () {
  // console.log('onLoad:' + app.globalData.domain)
    var that = this;
    // console.log(that.data.cardTeams)
    let hdlScore = "cardTeams["+ 0 +"].score"
    let xsqScore = "cardTeams["+ 1 +"].score"
    let tyScore  = "cardTeams["+ 2 +"].score"
    let njScore  = "cardTeams["+ 3 +"].score"
    let dldScore = "cardTeams["+ 4 +"].score"
    wx.request({
      //请求接口的地址
      url: 'http://10.108.217.31:7777/favorableRate', 
      method: "GET",
      header: {
        "Content-Type": "applciation/json" //默认值
        // "Content-Type": "application/x-www-form-urlencoded" //默认值
      },
      success: function (res) {
        //res相当于ajax里面的返回的数据
        console.log(res.data);
        var hdlScoreList = new Array()
        var xsqScoreList = new Array()
        var tyScoreList  = new Array()
        var njScoreList  = new Array()
        var dldScoreList = new Array()
        for (var key in res.data.markest_coarse_score.海底捞) {
          var item = res.data.markest_coarse_score.海底捞[key];
          hdlScoreList.push(item)
          // console.log(item);
        }
        // console.log(hdlScoreList)

        for (var key in res.data.markest_coarse_score.新石器烤肉) {
          var item = res.data.markest_coarse_score.新石器烤肉[key];
          xsqScoreList.push(item)
        }

        for (var key in res.data.markest_coarse_score.探鱼) {
          var item = res.data.markest_coarse_score.探鱼[key];
          tyScoreList.push(item)
        }

        for (var key in res.data.markest_coarse_score.南京大排档) {
          var item = res.data.markest_coarse_score.南京大排档[key];
          njScoreList.push(item)
        }

        for (var key in res.data.markest_coarse_score.多伦多海鲜自助) {
          var item = res.data.markest_coarse_score.多伦多海鲜自助[key];
          dldScoreList.push(item)
        }

        that.setData({
          [hdlScore]: res.data.market_final_score.海底捞,
          [xsqScore]: res.data.market_final_score.新石器烤肉,
          [tyScore]: res.data.market_final_score.探鱼,
          [njScore]: res.data.market_final_score.南京大排档,
          [dldScore]: res.data.market_final_score.多伦多海鲜自助,
        })
        app.globalData.hdlScoreList = hdlScoreList
        app.globalData.hdlAspectNegativeScoreList = res.data.market_aspect_count.海底捞.negative
        app.globalData.hdlAspectNeutralScoreList  = res.data.market_aspect_count.海底捞.neutral
        app.globalData.hdlAspectPositiveScoreList = res.data.market_aspect_count.海底捞.positive
        app.globalData.xsqScore = xsqScore
        app.globalData.xsqAspectNegativeScoreList = res.data.market_aspect_count.新石器烤肉.negative
        app.globalData.xsqAspectNeutralScoreList  = res.data.market_aspect_count.新石器烤肉.neutral
        app.globalData.xsqAspectPositiveScoreList = res.data.market_aspect_count.新石器烤肉.positive
        app.globalData.tyScore  = tyScore
        app.globalData.tyAspectNegativeScoreList = res.data.market_aspect_count.探鱼.negative
        app.globalData.tyAspectNeutralScoreList  = res.data.market_aspect_count.探鱼.neutral
        app.globalData.tyAspectPositiveScoreList = res.data.market_aspect_count.探鱼.positive
        app.globalData.njScore  = njScore
        app.globalData.njAspectNegativeScoreList = res.data.market_aspect_count.南京大排档.negative
        app.globalData.njAspectNeutralScoreList  = res.data.market_aspect_count.南京大排档.neutral
        app.globalData.njAspectPositiveScoreList = res.data.market_aspect_count.南京大排档.positive
        app.globalData.dldScore = dldScore
        app.globalData.dldAspectNegativeScoreList = res.data.market_aspect_count.多伦多海鲜自助.negative
        app.globalData.dldAspectNeutralScoreList  = res.data.market_aspect_count.多伦多海鲜自助.neutral
        app.globalData.dldAspectPositiveScoreList = res.data.market_aspect_count.多伦多海鲜自助.positive
      },
      fail: function (err) { },//请求失败
      complete: function () { }//请求完成后执行的函数
    })
  }
})