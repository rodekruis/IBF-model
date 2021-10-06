server <- function(input, output) {
  
  selected_pcode <- reactiveVal("KE011")
  #selected_pcode <- reactiveVal("LSA")
  #selected_indicator <- reactiveVal("spi3")
  #selected_indicator2 <- reactiveVal("spi6")
  
  
  Level <- reactive({
    req(input$Level)
    as.numeric(input$country)*as.numeric(input$Level)
  })
  
  
  country <- reactive({
    req(input$country)
    as.numeric(input$country)*as.numeric(input$Level)
  })
  
  
  Season_Obs_Rain <- reactive({
    req(input$Season_Obs_Rain)
    as.numeric(input$Season_Obs_Rain)
  })
  
  spi_index <- reactive({
    req(input$spi_index)
    as.character(input$spi_index)
  })
  
  
  
  climate_indicator_variable<- reactive({
    req(input$climate_indicator_variable)
    as.character(input$climate_indicator_variable)
  })
  

  rain_threshold <- reactive({
    req(input$rain_threshold)
    as.numeric(input$rain_threshold) 
  })
  
  spi_threshold <- reactive({
    req(input$spi_threshold)
    as.numeric(input$spi_threshold) 
  })
  
  
  SM_threshold <- reactive({
    req(input$SM_threshold)
    as.numeric(input$SM_threshold) 
  })
  
  
  vci_threshold<- reactive({
    req(input$vci_threshold)
    as.numeric(input$vci_threshold) 
  })
  
  

  
  
  Impact_df <- reactive({
    df_impact_raw[[country()]] %>%
      filter(
        pcode == selected_pcode()
      )
  })
  
 
  
  RAIN_PCODE2 <- reactive({
    req(input$Season_Obs_Rain)
    rainfall_sesonal_df %>%
      filter(ADM1_PCODE == selected_pcode(),
             season==isolate(input$Season_Obs_Rain)
             )
    })
  

  

  

  
  
  output$selected_district <- renderText({
    paste("Selected Aggregation area:  ", as.data.frame(admin[[country()]])[which(as.data.frame(admin[[country()]])[layerId[[country()]]] == selected_pcode()),label[[country()]]])
  })

  impact_df <- reactive({
    req(input$dateRange)
    df_impact_raw[[country()]] %>%
      filter(pcode == selected_pcode(),
             date >= isolate(input$dateRange[1]),
             date <= isolate(input$dateRange[2])
             )
  })
  
  
  output$trigger <- renderText({
    paste("Selected indicator:  ", climate_indicator_variable1[climate_indicator_variable()])
  })
  

 
 
  
  
# 
#   output$drought_indicators_plot <- renderPlotly({
#     req(input$Drought_indicator_variable)
#     req(input$climate_indicator_variable)
#     #req(input$spi_threshold)
#     #req(input$spi_threshold)
#     p <- plot_drought_indicators(
#       isolate(RAIN_PCODE()),
#       impact_df(),
#       input$Drought_indicator_variable,
#       input$climate_indicator_variable
#       )
#     p
#   })

  
  
  
  # output$drought_indicators_plot1 <- renderPlotly({
  #   req(input$spi_threshold)
  #   req(input$spi_index)
  #   
  #   p <- plot_matrix_spi(
  #     input$spi_index,
  #     input$spi_threshold,
  #     RAIN_PCODE()
  #   )
  #   p
  # })
  
 
  

  # output$drought_indicators_plot2 <- renderPlotly({
  #   req(input$vci_threshold)
  #   p <- plot_matrix_vci(
  #     input$vci_threshold,
  #     vci_PCODE()
  #   )
  #   p
  # })
  
  
  impact_hazard_sesonal<- reactive({
    req(input$Season_Obs_Rain)
    impact_hazard_sesonal_df %>%
      filter( season==isolate(input$Season_Obs_Rain),
              SPI==isolate(input$climate_indicator_variable)
      )
  })
  
  
 
  
  skill_table <- reactive({
    req(input$country)
    req(input$Level)
    req(input$Season_Obs_Rain) 
    req(input$SM_threshold) 
    req(input$vci_threshold) 
    req(input$spi_threshold)
    req(input$rain_threshold)
    req(input$climate_indicator_variable)
    isolate({
        
        predict_with_rain_skill(impact_hazard_sesonal_df,
                                SM_threshold(),
                                Season_Obs_Rain(),
                                vci_threshold(),
                                spi_threshold(),
                                rain_threshold(),
                                climate_indicator_variable(),
                                admin3=admin_lhz,
                                Level())
      })
  })
  
 
  
  
  

  output$skill_map <- renderLeaflet({
    pal <- colorBin("Greens", domain = skill_table()$CSI, bins = 4)
    #flood_palette <- colorNumeric(palette = "YlOrRd", domain = skill_table$FAR)
    leaflet() %>%
      addProviderTiles(providers$OpenStreetMap) %>%
      addPolygons( fillColor = ~ pal(CSI),
                  data = skill_table(), 
                  label=skill_table() %>% pull(CSI),#label[[3]]),
                  #layerId=skill_table() %>% pull(layerId[[3]]),
                  col=~pal(CSI), 
                  fillOpacity=0.8, 
                  opacity = 1,
                  weight=0.2)%>%leaflet::addLegend(pal = pal,values = col,title = "critical success index(CSI) for selected threshold",opacity = 0.7)
  })
    
    
  
  #addLegend("topright",pal = flood_palette,
  # values = admin[[country()]]$n_events,
  # title = "Reported Drought\n events (2000-2020)",
  #labFormat = labelFormat(prefix = "# reported floods "),
  # opacity = 1)
  
  
  

  

   
  
  
  # result_table <- reactive({predict_with_swi(all_days, swi_raw, df_impact_raw, selected_pcode(), input$swi_threshold)})
  
  result_table <- reactive({
    req(input$Season_Obs_Rain) 
    req(input$SM_threshold) 
    req(input$vci_threshold) 
    req(input$spi_threshold)
    req(input$rain_threshold)
    req(input$climate_indicator_variable)

     predict_with_rain(RAIN_PCODE2(),Season_Obs_Rain(),SM_threshold(),vci_threshold(),spi_threshold(),rain_threshold(),Impact_df(), climate_indicator_variable())

   })


  droughts_incorrect_val <- reactive({result_table() %>% pull(triggered_in_vain)})
  protocol_triggered_val <- reactive({result_table() %>% pull(triggered_correct)})
  detection_ratio_val <- reactive({result_table() %>% pull(POD)})
  false_alarm_ratio_val <- reactive({result_table() %>% pull(FAR)})

  output$result_html <- renderUI({
    HTML(
      paste0(
        '<span style="font-size:20px"> based on past impact data, how many time would we have acted in Vain: </span> ', droughts_incorrect_val(), "<br />",
        '<span style="font-size:20px"> based on past impact data, how many time would we have activated EAP correctly: </span> ', protocol_triggered_val(), "<br />",
        '<span style="font-size:20px"> Detection Ratio: </span> ', detection_ratio_val(), "<br />",
        '<span style="font-size:20px"> False Alarm Ratio: </span> ', false_alarm_ratio_val(), "<br />"
      )
    )
  })
  
 
  
  output$result_html2 <- renderUI({
    HTML(
      paste0(
        '<span style="color:blue;font-size:20px;font-weight:bold"> Critical Success Index(CSI) map for the selected drought indicator-threshold : </span> ',  "<br />")
    )
  })
  
  output$result_html3 <- renderUI({
    HTML(
      paste0(
        '<span style="color:red;font-size:20px;font-weight:bold"> Drought events frequency map based on reported drought events: </span> ',  "<br />")
    )
  })
  


  
 

  output$impact_map <- renderLeaflet({
    flood_palette <- colorNumeric(palette = "YlOrRd", domain = admin[[country()]]$n_events)
    flood_palette <-colorBin("YlOrRd", domain =admin[[country()]]$n_events, bins = 5)
    leaflet() %>%
      addProviderTiles(providers$OpenStreetMap) %>%
      addPolygons(data = admin[[country()]], label=admin[[country()]] %>% pull(label[[country()]]),
                  layerId=admin[[country()]] %>% pull(layerId[[country()]]),
                  col=~flood_palette(n_events), fillOpacity=0.8, opacity = 1, weight=1.2)%>%
      leaflet::addLegend(pal = flood_palette,values = col,title = "Reported Drought\n events (2000-2020)",opacity = 0.7)
      #addLegend("topright",pal = flood_palette,
               # values = admin[[country()]]$n_events,
               # title = "Reported Drought\n events (2000-2020)",
                #labFormat = labelFormat(prefix = "# reported floods "),
               # opacity = 1)
  })

  observeEvent(input$impact_map_shape_click, {  
    event <- input$impact_map_shape_click
    selected_pcode(event$id)
    cat(event$id)

    if (is.null(event$id)) { return() }  # Do nothing if it is a random click

  })
}