header <- dashboardHeader(
  title = "FBF Trigger dashboard",
  # Trick to put logo in the corner
  tags$li(div(
    class="logo_div",
    img(#src = 'https://www.510.global/wp-content/uploads/2017/07/510-LOGO-WEBSITE-01.png',#'510logo.png', title = "logo", height = "44px")
      src="https://data.humdata.org/image/2018-11-22-083630.964412EEE.PNG",
        title = "logo", height = "44px")),
    class = "dropdown")
)

ui_tab_main <- tabItem(
  "tab_main",
  fluidRow(
    column(
      width = 12,
	  uiOutput("result_html3"),
      leafletOutput("impact_map", height=400),
      h3(textOutput("selected_district")),
	  h3(textOutput("trigger")),
	  uiOutput("result_html2"),
	  column(width = 12,leafletOutput("skill_map", height=300)),
	  
    )
  ),
  fluidRow(
    column(
      width = 12,
      # sliderInput("ipc_plot", "Select SWI Threshold: ", min=10, max = 100, value=75),
      uiOutput("result_html"),
      #plotlyOutput("drought_indicators_plot1"),
      #plotlyOutput("drought_indicators_plot"),
      #plotlyOutput("drought_indicators_plot2"),
      #plotlyOutput("ipc_plot"),
     # uiOutput("indicaor2_slider")

      
    )

  )
)

body <- dashboardBody(
  # Loads CSS and JS from www/custom.css in
  tags$head(tags$link(rel = "stylesheet",
                      type = "text/css", href = "custom_css.css")),
  tags$head(tags$script(src="main.js")),
  tabItems(
    ui_tab_main
  )
)

ui <- dashboardPage(
  header,
  dashboardSidebar(
    collapsed=F,
    sidebarMenu(
      menuItem("Main Tab", tabName = "tab_main"),
      dateRangeInput('dateRange',
                     label = 'Select date range:',
                     start = min(df_impact_raw[[1]]$date, na.rm=T), end = max(df_impact_raw[[1]]$date, na.rm=T)),
      
      radioButtons("country", "Country:", c("Ethiopia" = 1)),
      #radioButtons("country", "Country:", c("Ethiopia" = 1, "Kenya" = 2, "Mozambique" = 3,"Lesotho" = 4,"Namibia" = 5)),
      selectInput("Level", "Select aggregation  Level(for now only Admin ):", c("Zones"=1, "Livelihood zones"=10),selected=1),#"LEVEL 3"),
      selectInput("Season_Obs_Rain", "Select rainfall season:", c("Belg" = 1,"Kirmet"=2),selected=1),
      selectInput("climate_indicator_variable", "Select meteorological indicator:", 
                  c("Sesonal Rain" = 'rain_season',"VCI"='vci', "TAMSAT_SM"='TAMSAT_sm',"SPI 1"='spi_1', "SPI 2"='spi_2',"SPI 3"='spi_3', "SPI 6"='spi_6',"SPI 12"='spi_12'),selected='rain_season'),
      #selectInput("spi_index", "Choose SPI index(months): ", c("SPI 1"=1, "SPI 2"=2,"SPI 3"=3, "SPI 6"=6,"SPI 12"=12),selected=1),#"LEVEL 3"),
      sliderInput("spi_threshold", "Select a Threshold for The SPI: ",max = 0, step=0.1, min = -1.5,value=-0.9),
      sliderInput("SM_threshold", "Select a Threshold for The TAMSAT SM: ",max = 1, step=0.1, min = 0,value=0.1),
      sliderInput("vci_threshold", "Select a Threshold for The vci: ",max = 50,step=5, min = 0,value=30),
      sliderInput("rain_threshold", "Select threshold for a reduction in % seasonal average rain: ",max = 50,step=10, min = 0,value=20)

      #selectInput("Drought_indicator_variable", "Select a biophysical Drought indicator:", c("VCI" = 'vci', "DMP" = 'vci'),selected='vci')
      )
  ),
  body
)