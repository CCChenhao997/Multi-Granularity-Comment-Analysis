//index.js
//获取应用实例
var app = getApp();
var cardTeams;
Page({
  data: {
    cardTeams: [{
      "viewid": "1",
      "imgsrc": "../../source/haidilao.jpeg",
      "name": "海底捞火锅",
      "url": "../visualization/visualization",
    }, 
    {
      "viewid": "2",
      "imgsrc": "../../source/kaorou.jpeg",
      "name": "新石器烤肉",
      "url": "../visualization/visualization",
    }, 
    {
      "viewid": "3",
      "imgsrc": "../../source/tanyu.jpeg",
      "name": "探鱼",
      "url": "../visualization/visualization",
    },
    {
      "viewid": "4",
      "imgsrc": "../../source/dapaidang.jpeg",
      "name": "南京大排档",
      "url": "../visualization/visualization",
    },
    {
      "viewid": "5",
      "imgsrc": "../../source/duolunduo.jpeg",
      "name": "多伦多海鲜自助",
      "url": "../visualization/visualization",
    }],
  },
  onLoad: function () {
  // console.log('onLoad:' + app.globalData.domain)
  },
})