//获取应用实例
var app = getApp()

Page({
  data: {
    tip: '',
    text: '',
    ishow: false,
    start: "选择餐馆",
    slist: [
      { id: 1, name: "海底捞" },
      { id: 1, name: "新石器烤肉" },
      { id: 1, name: "探鱼" },
      { id: 1, name: "南京大排档" },
      { id: 1, name: "多伦多海鲜自助" },
    ],
    isstart: false,
    openimg: "/source/emotion.png",
    offimg: "/source/person.png"
  },

  opens: function (e) {
    switch (e.currentTarget.dataset.item) {
     case "1":
      if (this.data.isstart) {
       this.setData({
        isstart: false,
        ishow: false,
       });
      }
      else {
       this.setData({
        isstart: true,
        ishow: false,
       });
      }
      break;
    }
   },

   onclicks1: function (e) {
    var index = e.currentTarget.dataset.index;
    let name = this.data.slist[index].name;
    this.setData({
     isstart: false,
     isfinish: false,
     isdates: false,
     start: this.data.slist[index].name,
     ishow: true,
     // finish: "目的地"
    })
    console.log(name)
   },

  formBindsubmit: function (e) {
    this.setData({
      // tip: '分析结果',
      text: e.detail.value.text,
    })
    this.getTableData();
  },

  formReset: function () {
    this.setData({
        tip: '',
        location: '',
        service: '',
        price: '',
        environment: '',
        dish: '',
        others: '',
        location_traffic_convenience: '',
        location_distance_from_business_district: '',
        location_easy_to_find: '',
        service_wait_time: '',
        service_waiters_attitude: '',
        service_parking_convenience: '',
        service_serving_speed: '',
        price_level: '',
        price_cost_effective: '',
        price_discount: '',
        environment_decoration: '',
        environment_noise: '',
        environment_space: '',
        environment_cleaness: '',
        dish_portion: '',
        dish_taste: '',
        dish_look: '',
        dish_recommendation: '',
        others_overall_experience: '',
        others_willing_to_consume_again: '',
    })
  },
  // onLoad: function () {
  //   this.getTableData();
  // }, /*onLoad-end */
  getTableData: function () {//自定义函数名称
    var that = this; 
    var resName = this.data.start
    // 这个地方非常重要，重置data{}里数据时候setData方法的this应为以及函数的this, 如果在下方的sucess直接写this就变成了wx.request()的this了
    wx.request({
       //请求接口的地址
      url: 'http://10.108.217.31:7777/sa', 
      data: {
        customerName: app.globalData.userInfo.nickName,
        customerGender: app.globalData.userInfo.gender,
        resName: resName,
        text: this.data.text,
      },
      method: "POST",
      header: {
        "Content-Type": "applciation/json" //默认值
        // "Content-Type": "application/x-www-form-urlencoded" //默认值
      },
      success: function (res) {
        //res相当于ajax里面的返回的数据
        console.log(res.data);
        //如果在sucess直接写this就变成了wx.request()的this了
        //必须为getTableData函数的this,不然无法重置调用函数
        that.setData({
          tip: '评分结果',
          datas: res.data,  //datas传值给页面的，可以自定义命名
          location: '位置: ' + res.data.Aspect_first_layer.location,
          service: '服务: ' + res.data.Aspect_first_layer.service,
          price: '价格: ' + res.data.Aspect_first_layer.price,
          environment: '环境: ' + res.data.Aspect_first_layer.environment,
          dish: '菜品: ' + res.data.Aspect_first_layer.dish,
          others: '其他: ' + res.data.Aspect_first_layer.others,
          location_traffic_convenience: '交通是否便利: ' + res.data.Aspect_second_layer.location_traffic_convenience,
          location_distance_from_business_district: '距离商圈远近: ' + res.data.Aspect_second_layer.location_distance_from_business_district,
          location_easy_to_find: '是否容易寻找: ' + res.data.Aspect_second_layer.location_easy_to_find,
          service_wait_time: '排队等候时间: ' + res.data.Aspect_second_layer.service_wait_time,
          service_waiters_attitude: '服务人员态度: ' + res.data.Aspect_second_layer.service_waiters_attitude,
          service_parking_convenience: '是否容易停车: ' + res.data.Aspect_second_layer.service_parking_convenience,
          service_serving_speed: '点菜/上菜速度: ' + res.data.Aspect_second_layer.service_serving_speed,
          price_level: '价格水平: ' + res.data.Aspect_second_layer.price_level,
          price_cost_effective: '性价比: ' + res.data.Aspect_second_layer.price_cost_effective,
          price_discount: '折扣力度: ' + res.data.Aspect_second_layer.price_discount,
          environment_decoration: '装修情况: ' + res.data.Aspect_second_layer.environment_decoration,
          environment_noise: '嘈杂情况: ' + res.data.Aspect_second_layer.environment_noise,
          environment_space: '就餐空间: ' + res.data.Aspect_second_layer.environment_space,
          environment_cleaness: '卫生情况: ' + res.data.Aspect_second_layer.environment_cleaness,
          dish_portion: '分量: ' + res.data.Aspect_second_layer.dish_portion,
          dish_taste: '口感: ' + res.data.Aspect_second_layer.dish_taste,
          dish_look: '外观: ' + res.data.Aspect_second_layer.dish_look,
          dish_recommendation: '推荐程度: ' + res.data.Aspect_second_layer.dish_recommendation,
          others_overall_experience: '本次消费感受: ' + res.data.Aspect_second_layer.others_overall_experience,
          others_willing_to_consume_again: '再次消费的意愿: ' + res.data.Aspect_second_layer.others_willing_to_consume_again
        })

        app.globalData.scoreList = res.data.scoreList
        app.globalData.aspectScore = res.data.radarScore
        // console.log(app.score_list)
        // console.log(app.aspectScore)
      },
      fail: function (err) { },//请求失败
      complete: function () { }//请求完成后执行的函数
    })
    
  },

  bindViewTab:function(){

    wx.navigateTo({    //保留当前页面，跳转到应用内的某个页面（最多打开5个页面，之后按钮就没有响应的）

         url:"/pages/visualization/visualization"

    })
  }
})
